"""An authored customer-support history, and the questions that grade a reader on it.

    from demo.scenario import conversation, questions   # or: sys.path.insert(0, "demo")

This is the corpus behind use case 01: *a support agent that must not contradict
itself*. The claim being tested is the customer's complaint, not ours — "the customer
already corrected this, and the agent said the old thing anyway" — and a corpus is only
worth running if it could come back and say the complaint stands.

## Why this exists rather than another public benchmark

Every retrieval number this project reports isolates architecture from model quality on
purpose, and none of them says whether an agent answering out of memvara *answers
better*. Measuring that needs a reader, and a reader needs questions with authored
answers. LOCOMO has those, and it has one defect this corpus does not: it is public, so
a hosted reader may have seen it, and a number that might be memorisation is not a
number. Nobody has seen these sixty-four turns. The trade is that they are synthetic and
were written by the same party that wrote the library — see `demo/README.md`, which
states that plainly, because it is the price of the number being worth anything.

## What the history contains, and why in this shape

One customer, a joiner in Sussex, from January to August 2026. Six facts move, and they
do not all move for the same reason — which is the distinction the whole library is
built on:

**The world changed** (valid time closes; memvara calls this `ended`). The plan went
Home, then Pro on 3 March, then Home again on 19 June. The delivery address changed when
they moved house on 26 July. The billing address followed it ten days later, on 5 August
— *not* the same day, which is the sharpest question in the set. The contact preference
was phone from 6 February and email from 22 June. Every one of these values was true
when it was recorded and is simply not true now.

**The record was wrong** (transaction time closes; `retired`). The mobile number taken
down on 6 February had two digits transposed, corrected on 13 February. The unit's
serial recorded in February was read off the power supply's label, corrected on 9 April.
Neither value was ever true. A corpus containing only the first kind cannot tell a
system that models both apart from one that models supersession alone, so both kinds are
here and the question set asks which is which.

Every question carries which of the two its fact moved on, in `Question.closure`, so a
results table can report the two failures separately. *Served a value that has expired*
and *served a value that was never true* are one column apart and are not the same
finding.

**Nothing changed.** The account name and the billing day are stated once and never
touched, so a system that reports change where there is none is caught.

**Never stated.** The Pro plan's monthly price and the identity of the card on file are
absent, and an honest reader has to say so.

## The one design decision that makes this corpus adversarial

A superseded value that is mentioned once, early, and never again is not the failure the
use case describes. Real support histories keep dragging the old value back into view —
customers complain about the invoice that went to the old address, and they read the
wrong sticker twice. So **every superseded value is re-surfaced late, in a past-tense or
mistaken framing**:

* the old address is the last address named in the whole transcript, and by then it has
  been mentioned nine times to the new address's three;
* the retired serial is the *last* serial a customer utters (6 August);
* the Pro plan is the *last* plan name in the transcript (6 August, past tense);
* the turn that reverses the contact preference (22 June) states the superseded one
  inside itself — "I know I asked for calls back in February" — so the newest turn on
  that subject is also the one that says the old answer out loud.

Recency and emphasis both point at the wrong answer. That is what makes a wrong answer
here evidence of something rather than noise.

## Contract

`conversation()` and `questions()` are pure and deterministic — no clock, no randomness,
no I/O — because everything downstream compares runs to each other.

>>> turns = conversation()
>>> len(turns), turns[0].role, turns[0].at.isoformat()
(64, 'user', '2026-01-19T09:12:00+00:00')
>>> conversation() == conversation()
True
>>> qs = questions()
>>> len(qs), sorted({q.kind for q in qs})
(20, ['correction', 'current', 'historical', 'unanswerable'])
>>> sorted({q.closure for q in qs if q.closure})
['ended', 'retired']
>>> next(q for q in qs if q.id == "q_billing_address_on_30_july").about
datetime.datetime(2026, 7, 30, 12, 0, tzinfo=datetime.timezone.utc)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

__all__ = ["Turn", "Question", "conversation", "questions"]

UTC = timezone.utc


def _at(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """A tz-aware UTC instant, spelled short enough that the turn table stays readable.

    UTC throughout, including for a customer in the UK who is on BST for most of this
    history. Storing local time would make the corpus a timezone exercise, and the one
    thing every consumer of these timestamps needs is that they are comparable.

    >>> _at(2026, 3, 3, 11, 5).isoformat()
    '2026-03-03T11:05:00+00:00'
    """
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


@dataclass(frozen=True)
class Turn:
    """One utterance in the support history.

    `role` is `"user"` for the customer and `"assistant"` for the support agent, in the
    reader model's vocabulary rather than the support desk's, because a reader is asked
    to *be* the agent.
    """

    #: When the turn happened. Tz-aware UTC.
    at: datetime
    #: `"user"` (the customer) or `"assistant"` (the support agent).
    role: str
    text: str


@dataclass(frozen=True)
class Question:
    """One question with an authored answer, and the wrong answer worth counting.

    `gold` was written from the transcript by hand. It is deliberately *not* whatever
    memvara returns: an answer key derived from the system under test measures nothing.
    Every gold below carries a comment naming the turn that justifies it, and
    `tests/test_demo_scenario.py` pins that link so a later edit to the history that
    invalidates an answer fails a test instead of quietly rescoring the run.

    `trap` is the specific wrong answer a system with no bitemporal handling produces —
    almost always the superseded or retracted value, because this corpus states those
    with more emphasis and more recently than the standing ones. Reporting "how many
    were right" and "how many gave the superseded answer" as two numbers is the point:
    the second is what the marketing claim actually rests on. Where a wrong answer could
    be anything, `trap` is `None` rather than a guess — a made-up trap would inflate the
    interesting number with questions that never had a single failure mode.

    **Scoring note.** For `kind="correction"` the gold necessarily contains the trap,
    because the answer's whole job is to name both values. A grader that scores "gave
    the trap" by substring containment will therefore mark every correct correction
    answer as a trap hit. Score those on whether the standing value is present and
    labelled as standing; see `demo/README.md`.
    """

    #: Stable across edits to the history. Results tables key on this.
    id: str
    #: When the question is put. Tz-aware UTC. Usually after every turn, but not always:
    #: two questions are asked mid-history so that a system which peeks at the future
    #: can be caught. `kind` is relative to *this* instant, not to today.
    asked_at: datetime
    text: str
    #: The correct answer, authored from the transcript.
    gold: str
    #: The superseded or retracted value a naive system returns, or `None` when the
    #: failure mode is diffuse.
    trap: str | None
    #: `"current"` | `"historical"` | `"correction"` | `"unanswerable"`.
    kind: str
    #: Which clock closed on this question's fact, in `memvara.types.Closure`'s
    #: vocabulary. `"ended"` where the value that is not the standing one was closed on
    #: **valid** time — the world changed, and it was true when it was recorded.
    #: `"retired"` where it was closed on **transaction** time — the record was wrong and
    #: was never true at all. `None` where neither applies: the controls, where nothing
    #: has been closed on either axis; the two summary questions, which span both and
    #: would be misdescribed by either; and the unanswerables, which have no fact.
    #:
    #: Orthogonal to `kind`, and that is the point. `q_serial_current` is
    #: `kind="current", closure="retired"` and `q_serial_correction` is
    #: `kind="correction", closure="retired"` — the same wrong record, asked two ways —
    #: while `q_plan_current` is `kind="current", closure="ended"`. Without this field a
    #: results table has one trapped rate covering two failures that mean opposite
    #: things: *we served a value that has expired* and *we served a value that was never
    #: true*. The first is a stale cache. The second is the thing the library exists to
    #: prevent, and merging them into one percentage throws away the distinction the
    #: whole model is built on.
    closure: str | None = None
    #: The **valid-time** instant the question asks about — what `valid_at=` would be set
    #: to in a `search()` that answered it automatically. Left `None` where the question
    #: is about the moment it is put, since `asked_at` already carries that, and where no
    #: single instant is being asked about.
    #:
    #: Necessarily `None` on every `closure="retired"` question, and not for want of
    #: writing one down: a retracted value was never true at any world-time, so there is
    #: no valid-time instant at which the corpus said it. That is what makes a retraction
    #: a transaction-time event — `known_at` is the axis those questions move on, and it
    #: is not this field.
    about: datetime | None = None


# --- the values that move, written once ------------------------------------------
# Questions reference these; the turns spell them out in prose, the way a person would.
# Two spellings of one address is how a corpus develops an answer key that disagrees
# with itself, so the prose is written against these constants and the grounding test in
# `tests/test_demo_scenario.py` checks that each one is actually somewhere in the turns.

#: Delivery and billing address until the move. Nine mentions against the new address's
#: three, and it is the last address anyone names — which is what makes it the trap for
#: both "where do we ship now" and "where do invoices go now".
_OLD_ADDRESS = "41 Coldharbour Road, Lewes, BN7 2GT"
#: Delivery address from 26 July 2026, billing address from 5 August 2026.
_NEW_ADDRESS = "Bramble Cottage, Ditchling Road, Westmeston, BN6 8XA"

#: Recorded 6 February, retired 9 April: it is the number printed on the power supply,
#: and was never this unit's serial. Still the last serial number a customer says out
#: loud - on 6 August, as "the 4419 one", four months after correcting it.
_WRONG_SERIAL = "HX2-4419-B"
#: The unit's serial, and always was.
_SERIAL = "HX7-8802-D"

#: Recorded 6 February, retired 13 February: two digits transposed by the customer.
_WRONG_MOBILE = "07700 900 118"
_MOBILE = "07700 900 811"


# --- the history -----------------------------------------------------------------


_CONVERSATION: tuple[Turn, ...] = (
    # === 19 January 2026 - new install, poor coverage in the workshop ==============
    # Establishes three things: the account name (a control - never changes), the
    # delivery address, and the Home plan the account starts on.
    Turn(
        _at(2026, 1, 19, 9, 12),
        "user",
        "Morning. I got the HX7 kit on Friday and it's been running since Saturday, "
        "but the workshop end of it is hopeless. Two bars in the house and nothing at "
        "all past the back door.",
    ),
    Turn(
        _at(2026, 1, 19, 9, 19),
        "assistant",
        "Morning Dara, and thanks for the detail. Before we look at coverage, can I "
        "confirm this is the account under Wray & Daughter Joinery? There are two "
        "Wrays on the system and I want to be sure I'm in the right one.",
    ),
    Turn(
        _at(2026, 1, 19, 9, 22),
        "user",
        "That's the one, Wray & Daughter Joinery. It's the business account, my "
        "accountant set it up that way so the VAT works out. Everything comes to 41 "
        "Coldharbour Road, Lewes, BN7 2GT. House and workshop are the same address, "
        "the workshop is the block at the end of the garden.",
    ),
    Turn(
        _at(2026, 1, 19, 9, 26),
        "assistant",
        "Got it. You're on the Home plan, which covers the two nodes you have. The "
        "garden block is going to be the problem, though. How far is it from the "
        "house, roughly?",
    ),
    Turn(
        _at(2026, 1, 19, 9, 29),
        "user",
        "Thirty metres give or take, and there's a brick wall in between. It was a "
        "coal store before I converted it, so the walls are a foot thick.",
    ),
    Turn(
        _at(2026, 1, 19, 9, 34),
        "assistant",
        "Then two nodes will not reach it reliably, no matter where we put them. Try "
        "moving the second node to the back bedroom window tonight, on the wall facing "
        "the garden, and tell me what you see out there tomorrow. If that gets you one "
        "bar we've got something to work with.",
    ),
    Turn(
        _at(2026, 1, 19, 9, 41),
        "user",
        "Fine, I'll try it tonight. It isn't urgent urgent. I just can't take card "
        "payments out there and people turn up wanting to pay for a table.",
    ),
    # === 6 February 2026 - dropouts. The two records that turn out to be wrong ======
    # The serial and the mobile are both taken down here, and both are wrong. Nothing
    # in this ticket says so; that is the point. A support desk records what it is
    # told, and finds out later.
    Turn(
        _at(2026, 2, 6, 16, 2),
        "user",
        "It's dropping out. Not the workshop this time, the whole lot. Everything in "
        "the house went off about half three, came back on its own, then went again "
        "ten minutes later. My card reader lost a payment mid transaction.",
    ),
    Turn(
        _at(2026, 2, 6, 16, 8),
        "assistant",
        "Sorry about that. I can pull the logs, but I need the serial off the unit "
        "first. It'll be on a sticker.",
    ),
    Turn(
        _at(2026, 2, 6, 16, 14),
        "user",
        "HX2-4419-B.",
    ),
    Turn(
        _at(2026, 2, 6, 16, 17),
        "assistant",
        "Thank you. And the best number for you? If the logs show a hardware fault I'd "
        "rather ring you than go back and forth over email.",
    ),
    Turn(
        _at(2026, 2, 6, 16, 21),
        "user",
        "07700 900 118. And please do ring. Don't email me. I'm in the workshop from "
        "seven and I've usually got the saw or the planer going. I don't look at email "
        "until the evening and by then it's a day gone. Phone or a text, either one, "
        "but not email.",
    ),
    Turn(
        _at(2026, 2, 6, 16, 29),
        "assistant",
        "Noted, and I've set the contact preference on the account to phone. I'll flag "
        "this ticket for a callback rather than an email reply.",
    ),
    Turn(
        _at(2026, 2, 6, 16, 35),
        "user",
        "Appreciated.",
    ),
    # === 13 February 2026 - correction one: the record was wrong ====================
    # The customer's own framing is what makes this a retraction rather than a change:
    # "it's been wrong since I gave it to you". No number changed hands. The February
    # record was never true, which is exactly what `close="retired"` means.
    Turn(
        _at(2026, 2, 13, 8, 40),
        "user",
        "Did someone try to ring me yesterday? Nothing came through. Hang on though. "
        "The number I gave you last week is wrong, I've just read it back off your "
        "email. I said 07700 900 118 and it's 07700 900 811. I've gone and transposed "
        "it. That's my fault, and it's been wrong since the moment I gave it to you, "
        "so whoever's been ringing has been ringing a stranger.",
    ),
    Turn(
        _at(2026, 2, 13, 8, 47),
        "assistant",
        "That would explain it. I've put 07700 900 811 on the account as a correction "
        "rather than as a new number, so the record doesn't claim the old one was ever "
        "yours.",
    ),
    Turn(
        _at(2026, 2, 13, 8, 52),
        "user",
        "Good. Sorry about that.",
    ),
    # === 3 March 2026 - the upgrade. The world changed. =============================
    Turn(
        _at(2026, 3, 3, 11, 5),
        "user",
        "I need more out of this. We've taken the second bench at Halden Yard, signed "
        "the lease Friday, and I need the till and the CNC on the same network as the "
        "house. My brother in law does the books remotely and he keeps getting kicked "
        "off as well.",
    ),
    Turn(
        _at(2026, 3, 3, 11, 12),
        "assistant",
        "That's the Pro plan. Three more nodes on top of your two, the static IP your "
        "till wants, and priority support, which in practice means you get a person "
        "rather than a queue.",
    ),
    Turn(
        _at(2026, 3, 3, 11, 18),
        "user",
        "Do it. From today if you can.",
    ),
    Turn(
        _at(2026, 3, 3, 11, 24),
        "assistant",
        "Done. The account is on Pro as of today, 3 March, and the three extra nodes "
        "go out this afternoon.",
    ),
    Turn(
        _at(2026, 3, 3, 11, 31),
        "user",
        "Ship them to Coldharbour Road, not the Yard. There's nobody at the Yard to "
        "sign for anything until the benches are in.",
    ),
    Turn(
        _at(2026, 3, 3, 11, 40),
        "assistant",
        "Understood, they'll go to 41 Coldharbour Road.",
    ),
    # === 17 March 2026 - noise, with a distractor in it =============================
    # A retriever that pulls this for "how should we contact them" or "what name is the
    # account in" is being fooled by vocabulary. Real histories are full of turns like
    # this and a corpus with none of them is easier than the job.
    Turn(
        _at(2026, 3, 17, 12, 2),
        "user",
        "Did one of yours ring me on Sunday? Came up as a mobile, asked me to confirm "
        "the name on the account and I put the phone down on them.",
    ),
    Turn(
        _at(2026, 3, 17, 12, 9),
        "assistant",
        "Not us. We're closed Sundays and we'd never ring and ask you to confirm the "
        "account name, we'd already have it. You did the right thing.",
    ),
    # === 9 April 2026 - correction two, plus the price that becomes a distractor =====
    Turn(
        _at(2026, 4, 9, 13, 20),
        "user",
        "One of the new nodes has died. The one at the Yard. No lights at all, and "
        "I've tried a different socket and a different cable.",
    ),
    Turn(
        _at(2026, 4, 9, 13, 26),
        "assistant",
        "I'll get it swapped. Can you read me the serial off the dead one? And while "
        "I'm in here, the serial on the account for the main unit is HX2-4419-B. Is "
        "that still right?",
    ),
    Turn(
        _at(2026, 4, 9, 13, 38),
        "user",
        "Hang on, I'm under the bench. The dead node is HX7-6120-N. And that other "
        "number, where did you get 4419 from? Oh. That's off the power brick, the "
        "black plug thing, not the unit. I must have read you the wrong sticker in "
        "February. The unit itself says HX7-8802-D underneath. So 4419 has never been "
        "the unit's number at all, it's the power supply's.",
    ),
    Turn(
        _at(2026, 4, 9, 13, 47),
        "assistant",
        "Thank you, that's worth catching. I've put HX7-8802-D on the account as a "
        "correction to a misread label, not as a replacement unit. If I'd logged it as "
        "a swap your warranty would have restarted today and you'd have lost the two "
        "months you've already had.",
    ),
    Turn(
        _at(2026, 4, 9, 13, 52),
        "user",
        "Right. How long for the replacement node?",
    ),
    Turn(
        _at(2026, 4, 9, 13, 58),
        "assistant",
        "Two working days, out to 41 Coldharbour Road. If you'd rather have a spare on "
        "the shelf than wait next time, an extra node on top of the ones you have is "
        "£79.",
    ),
    Turn(
        _at(2026, 4, 9, 14, 5),
        "user",
        "Not at seventy nine quid it isn't. Just the replacement, thanks.",
    ),
    # === 21 May 2026 - billing. Establishes the billing address and a second control ==
    # The billing address is recorded here as a fact in its own right, which is what
    # makes 26 July - 5 August a window rather than an oversight.
    Turn(
        _at(2026, 5, 21, 10, 2),
        "user",
        "Two things. My accountant wants the invoices as PDF attachments rather than "
        "links, and he's asking what date this comes out of the account.",
    ),
    Turn(
        _at(2026, 5, 21, 10, 9),
        "assistant",
        "Billing is the 14th of the month, every month, and that date doesn't move. "
        "I've switched the invoices to PDF attachments from the next one.",
    ),
    Turn(
        _at(2026, 5, 21, 10, 14),
        "user",
        "And where do they go? He wants the paper ones as well, he's old fashioned.",
    ),
    Turn(
        _at(2026, 5, 21, 10, 20),
        "assistant",
        "Paper invoices go to the billing address, which is 41 Coldharbour Road, "
        "Lewes, BN7 2GT, the same as delivery.",
    ),
    Turn(
        _at(2026, 5, 21, 10, 30),
        "user",
        "Grand. That's him off my back.",
    ),
    # === 19 June 2026 - the downgrade. The world changed again. =====================
    Turn(
        _at(2026, 6, 19, 15, 41),
        "user",
        "We've lost the Yard. Landlord's selling the whole block. Everything's back in "
        "the garden workshop by the end of the month, so I don't need the Pro thing "
        "any more. The static IP was only ever for the till.",
    ),
    Turn(
        _at(2026, 6, 19, 15, 49),
        "assistant",
        "Sorry to hear it. I can put you back on Home. You'd keep two nodes covered "
        "and the other three stop being supported, though they'll carry on working "
        "until they don't.",
    ),
    Turn(
        _at(2026, 6, 19, 15, 55),
        "user",
        "Do it. Today, if it saves me a month.",
    ),
    Turn(
        _at(2026, 6, 19, 16, 2),
        "assistant",
        "Done. The account is back on Home from today, 19 June. Pro ran from 3 March "
        "until today.",
    ),
    Turn(
        _at(2026, 6, 19, 16, 10),
        "user",
        "Thanks. Sorry, it's been a week.",
    ),
    # === 22 June 2026 - the preference reverses. Also the world changing. ============
    # The customer names the earlier preference and says why it no longer holds, which
    # is what a change looks like from the inside. Nobody made a mistake in February.
    Turn(
        _at(2026, 6, 22, 7, 55),
        "user",
        "Can you stop ringing me. I know I asked for calls back in February, and that "
        "was right at the time, I was on my own out there. Now I've got the tablet in "
        "the workshop with the email on it, and when you ring I've got the extractor "
        "going and I miss it anyway. Email from now on please.",
    ),
    Turn(
        _at(2026, 6, 22, 8, 4),
        "assistant",
        "Understood. I've changed the contact preference on the account to email from "
        "today.",
    ),
    Turn(
        _at(2026, 6, 22, 8, 12),
        "user",
        "Thank you.",
    ),
    Turn(
        _at(2026, 6, 22, 8, 20),
        "assistant",
        "Anything else while you're here?",
    ),
    # === 26 July 2026 - the move. Delivery changes. Billing does not. ================
    Turn(
        _at(2026, 7, 26, 18, 22),
        "user",
        "We've moved. Not the workshop, that's a summer's job, but the house has gone. "
        "As of Friday everything comes to Bramble Cottage, Ditchling Road, Westmeston, "
        "BN6 8XA. Coldharbour's sold, there's a family in it now, so please don't send "
        "anything there.",
    ),
    Turn(
        _at(2026, 7, 26, 18, 31),
        "assistant",
        "Congratulations, and noted. I've changed the delivery address to Bramble "
        "Cottage, Ditchling Road, Westmeston, BN6 8XA.",
    ),
    Turn(
        _at(2026, 7, 26, 18, 38),
        "user",
        "The wifi here is dreadful, by the way, but that's tomorrow's problem. The "
        "cottage walls are two foot of flint.",
    ),
    Turn(
        _at(2026, 7, 26, 18, 45),
        "assistant",
        "Shout when you want to look at it. My guess is you'll want a third node, and "
        "flint is about the worst thing you can put between two of them.",
    ),
    Turn(
        _at(2026, 7, 26, 18, 50),
        "user",
        "Probably. Not today, I'm sat on a box.",
    ),
    # === 5 August 2026 - the invoice goes to the old address =========================
    # The most recent mention of any address in this history is the *old* one, twice,
    # in a complaint. A system ranking by recency ships the next node to a family who
    # bought the house in July.
    Turn(
        _at(2026, 8, 5, 9, 15),
        "user",
        "The July invoice has gone to Coldharbour Road. The new people put it through "
        "my mother's door because she's two streets over and they know her. I told you "
        "we'd moved.",
    ),
    Turn(
        _at(2026, 8, 5, 9, 24),
        "assistant",
        "You did, and I'm sorry. I changed the delivery address on the 26th, but "
        "invoices go to a separate billing address and that one was still 41 "
        "Coldharbour Road until this morning. I've changed it now to Bramble Cottage, "
        "Ditchling Road, Westmeston, BN6 8XA, so August's will come to the right door.",
    ),
    Turn(
        _at(2026, 8, 5, 9, 31),
        "user",
        "Two addresses. Of course there are two addresses.",
    ),
    Turn(
        _at(2026, 8, 5, 9, 38),
        "assistant",
        "There are, and yours matched each other until you moved, which is why nobody "
        "noticed. One more thing while we're here: the card on file still has the old "
        "billing address against it at your bank's end, and that can bounce the "
        "payment on the 14th. Can you update it with them?",
    ),
    Turn(
        _at(2026, 8, 5, 9, 46),
        "user",
        "I'll do it tonight.",
    ),
    Turn(
        _at(2026, 8, 5, 9, 52),
        "assistant",
        "Thanks. If it does bounce we'll email you rather than ring.",
    ),
    # === 6 August 2026 - the last word, and it is the wrong one twice ================
    # The final plan name in the transcript is Pro, past tense. The final serial a
    # customer says is the retired one. Both are natural things for a person to say and
    # both are the trap.
    Turn(
        _at(2026, 8, 6, 19, 40),
        "user",
        "Card's sorted. The people at Coldharbour Road put another envelope through my "
        "mother's door yesterday, by the way. Not one of yours, it's the council. And "
        "I've been thinking about the cottage: it's worse in here than the workshop "
        "ever was, I get nothing upstairs at all.",
    ),
    Turn(
        _at(2026, 8, 6, 19, 47),
        "assistant",
        "On Home you're covered for the two nodes. A third is an add on and it's the "
        "same £79 it was in April.",
    ),
    Turn(
        _at(2026, 8, 6, 19, 53),
        "user",
        "Back in the spring when I was on Pro I had five of the things running and the "
        "far end of the Yard still dropped, so I'm not convinced more boxes is the "
        "answer. Email me the options and I'll look at the weekend.",
    ),
    Turn(
        _at(2026, 8, 6, 19, 58),
        "assistant",
        "I'll send them over tonight.",
    ),
    Turn(
        _at(2026, 8, 6, 20, 5),
        "user",
        "One more thing. I'm doing the insurance schedule and it wants the serial. Is "
        "it the 4419 one? That's the number I can see from here.",
    ),
    Turn(
        _at(2026, 8, 6, 20, 11),
        "assistant",
        "That's the power supply again. The unit's serial is HX7-8802-D, and that's "
        "the one the insurer wants.",
    ),
    Turn(
        _at(2026, 8, 6, 20, 16),
        "user",
        "It's the same sticker I keep finding. Thanks.",
    ),
)


# --- the questions ---------------------------------------------------------------
# Grouped by kind for reading; the order below is the order they are asked in.
#
# `_ASKED` is the instant almost every question is put: a week after the last turn, from
# a support desk with the whole history behind it. The two exceptions are asked mid
# history on purpose and say why in place.

_ASKED = _at(2026, 8, 13, 9, 0)


_QUESTIONS: tuple[Question, ...] = (
    # --- current: what is true at `asked_at` --------------------------------------
    # The trap is the superseded value in every one of these. Note the direction: for
    # `current` questions the trap is the *old* value, and for `historical` questions
    # below it is the *current* one. A system that always answers with the most recent
    # thing it retrieved fails one group; a system that always answers with the most
    # emphatic fails the other.
    Question(
        id="q_plan_current",
        asked_at=_ASKED,
        text="The customer is on the phone asking what they are paying for. Which "
             "plan is this account on today?",
        # Justified by 19 June 16:02: "back on Home from today, 19 June". The trap is
        # justified by 6 August 19:53, where Pro is the last plan named in the whole
        # transcript - in the past tense, which is the part a naive reader drops.
        gold="Home.",
        trap="Pro",
        kind="current",
        closure="ended",
    ),
    Question(
        id="q_shipping_current",
        asked_at=_ASKED,
        text="A replacement node has to go out today. What address should it ship to?",
        # Justified by 26 July 18:31. The trap is justified by 5 August 9:15, a
        # complaint about the old address, and 6 August 19:40, which is the last time
        # any address is named at all - against 26 July 18:22, where the customer says
        # outright that a family lives at the old one now.
        gold=_NEW_ADDRESS,
        trap=_OLD_ADDRESS,
        kind="current",
        closure="ended",
    ),
    Question(
        id="q_billing_current",
        asked_at=_ASKED,
        text="Where are this account's paper invoices sent now?",
        # Justified by 5 August 9:24: billing changed to the new address that morning.
        gold=_NEW_ADDRESS,
        trap=_OLD_ADDRESS,
        kind="current",
        closure="ended",
    ),
    Question(
        id="q_contact_current",
        asked_at=_ASKED,
        text="We need to reach this customer about a service window. How do they want "
             "to be contacted?",
        # Justified by 22 June 7:55 and 8:04. The gold deliberately avoids the words in
        # the trap so a containment-based grader cannot score one as the other.
        gold="By email.",
        trap="By phone or text",
        kind="current",
        closure="ended",
    ),
    Question(
        id="q_serial_current",
        asked_at=_ASKED,
        text="What is the serial number of this customer's main unit?",
        # Justified by 9 April 13:38 and restated 6 August 20:11. The trap is justified
        # by 6 August 20:05, where the customer offers the retired number again - the
        # last serial any customer turn contains.
        gold=_SERIAL,
        trap=_WRONG_SERIAL,
        kind="current",
        closure="retired",
    ),
    Question(
        id="q_mobile_current",
        asked_at=_ASKED,
        text="What mobile number should be on file for this customer?",
        # Justified by 13 February 8:40 and 8:47.
        gold=_MOBILE,
        trap=_WRONG_MOBILE,
        kind="current",
        closure="retired",
    ),
    Question(
        id="q_account_name",
        asked_at=_ASKED,
        text="What name is this account registered in?",
        # The control. Stated once, 19 January 9:22, and never touched again. Nothing
        # supersedes it, so there is no trap: a wrong answer here is an extraction or
        # retrieval failure, not a time-handling one, and counting it as a trap hit
        # would put noise in the number the claim rests on.
        gold="Wray & Daughter Joinery.",
        trap=None,
        kind="current",
        # Nothing has been closed on this fact on either axis, which is what a
        # control is. `closure=None` and `trap=None` travel together here.
        closure=None,
    ),
    Question(
        id="q_billing_day",
        asked_at=_ASKED,
        text="Which day of the month is this account billed on?",
        # The second control. Stated once, 21 May 10:09, with "that date doesn't move".
        gold="The 14th of the month.",
        trap=None,
        kind="current",
        closure=None,
    ),
    # Asked mid-history. `demo/baselines.py` gives every arm the same cutoff, so these
    # two are answered from a truncated history, and a system that reaches past
    # `asked_at` is caught rather than rewarded.
    Question(
        id="q_plan_current_in_april",
        asked_at=_at(2026, 4, 20, 9, 0),
        # Byte-identical to `q_plan_current`, and pinned that way by
        # `test_the_matched_pairs_differ_only_in_when_they_are_asked`. A matched control
        # that is also reworded is two changes at once, and neither could be blamed.
        text="The customer is on the phone asking what they are paying for. Which "
             "plan is this account on today?",
        # Justified by 3 March 11:24. This is the matched control for
        # `q_plan_current`: same wording, same fact, an ask instant where the answer is
        # the opposite value. A system that gets this right and `q_plan_current` wrong
        # has a time problem, not an extraction problem, and the pair is what makes
        # that distinction available in the results table.
        gold="Pro.",
        trap="Home",
        kind="current",
        closure="ended",
    ),
    Question(
        id="q_serial_current_in_may",
        asked_at=_at(2026, 5, 1, 9, 0),
        text="What is the serial number of this customer's main unit?",
        # Justified by 9 April 13:38. Same question as `q_serial_current` without the
        # 6 August turn where the customer offers the retired number again, so the pair
        # isolates exactly what that late re-surfacing costs.
        gold=_SERIAL,
        trap=_WRONG_SERIAL,
        kind="current",
        closure="retired",
    ),
    # --- historical: what was true at a named past instant -------------------------
    Question(
        id="q_plan_mid_march",
        asked_at=_ASKED,
        text="Which plan was this account on in the middle of March 2026?",
        # Justified by 3 March 11:24 and 19 June 16:02, which together bound Pro. The
        # trap is the current value: this is the failure of a store that keeps only the
        # latest value per field, which is every store without valid time.
        gold="Pro.",
        trap="Home",
        kind="historical",
        closure="ended",
        # Mid-March, inside the Pro interval that runs 3 March to 19 June. Any
        # instant in that window answers the same, which is what makes the
        # question fair to a system reading the prose rather than this field.
        about=_at(2026, 3, 15, 12, 0),
    ),
    Question(
        id="q_shipping_april",
        asked_at=_ASKED,
        text="A replacement node was posted out in April 2026. Which address did it "
             "go to?",
        # Justified by 9 April 13:58, verbatim.
        gold=_OLD_ADDRESS,
        trap=_NEW_ADDRESS,
        kind="historical",
        closure="ended",
        # The day the parcel was addressed, not the day it landed: which address
        # was in force is a property of the former.
        about=_at(2026, 4, 9, 14, 0),
    ),
    Question(
        id="q_billing_address_on_30_july",
        asked_at=_ASKED,
        text="On 30 July 2026, which address did this account have on file for "
             "invoices?",
        # The sharpest question here. Justified by 5 August 9:24, where the agent says
        # billing was still the old address "until this morning" - so on 30 July the
        # delivery address had moved and the billing address had not. The trap is what
        # a system returns if it treats the move as one event: the two addresses were
        # the same fact for six months and then were not, for ten days.
        gold=_OLD_ADDRESS,
        trap=_NEW_ADDRESS,
        kind="historical",
        closure="ended",
        # Inside the ten-day window: after the delivery address moved on 26 July,
        # before the billing address followed on 5 August.
        about=_at(2026, 7, 30, 12, 0),
    ),
    Question(
        id="q_contact_april",
        asked_at=_ASKED,
        text="In April 2026, how had this customer asked us to get in touch?",
        # Justified by 6 February 16:21 and 16:29, which stood until 22 June.
        gold="By phone or text.",
        trap="By email",
        kind="historical",
        closure="ended",
        about=_at(2026, 4, 15, 12, 0),
    ),
    Question(
        id="q_plan_history",
        asked_at=_ASKED,
        text="Give the full plan history for this account, with dates.",
        # Justified by 19 January 9:26, 3 March 11:24 and 19 June 16:02. No trap: the
        # likely failure is collapsing the history to one value, and every value in the
        # collapse also appears in the gold, so a trap here would be unscoreable by
        # containment and a guess by any other method.
        gold="Home from the start, Pro from 3 March 2026, and Home again from "
             "19 June 2026.",
        trap=None,
        kind="historical",
        # The one historical question with no `about`, and not an oversight: it
        # asks for the whole sequence, so there is no single valid-time instant to
        # name and inventing one would make the field a lie. The exemption is pinned
        # by id in `test_the_only_historical_question_without_an_about_...`, so a new
        # historical question that forgets `about` still fails.
        closure=None,
        about=None,
    ),
    # --- correction: the record was wrong, and the answer has to say so ------------
    # These are the questions no store without transaction time can answer at all. A
    # superseded value and a retracted value look identical in a store with one clock:
    # both are "the old value". Here the honest answer is that one of them was never a
    # value. Note that gold contains trap by construction - see `Question`.
    Question(
        id="q_serial_correction",
        asked_at=_ASKED,
        text="The insurer is asking for the unit's serial number and whether it has "
             "ever changed. What do we tell them?",
        # Justified by 9 April 13:38 and 13:47, including the warranty consequence,
        # which is the practical reason the distinction is not pedantry.
        gold="It has never changed. The serial is HX7-8802-D and always has been. The "
             "account briefly recorded HX2-4419-B, which was a misreading of the "
             "label on the power supply, so no unit was ever replaced and the "
             "warranty still runs from the original purchase.",
        trap=_WRONG_SERIAL,
        kind="correction",
        closure="retired",
    ),
    Question(
        id="q_mobile_correction",
        asked_at=_ASKED,
        text="Did this customer change their mobile number, or did we record it "
             "wrongly? Give both numbers.",
        # Justified by 13 February 8:40, where the customer says it had been wrong from
        # the moment they gave it.
        gold="We recorded it wrongly. The customer transposed two digits on 6 "
             "February and corrected it on 13 February: 07700 900 118 was never their "
             "number, and 07700 900 811 has been their number throughout.",
        trap=_WRONG_MOBILE,
        kind="correction",
        closure="retired",
    ),
    Question(
        id="q_which_were_corrections",
        asked_at=_ASKED,
        text="Which details on this account were recorded wrongly and later fixed, as "
             "opposed to details that genuinely changed?",
        # The question the library exists for. Justified by 13 February 8:40 and 9
        # April 13:38 for the corrections, and by 3 March, 19 June, 22 June, 26 July
        # and 5 August for the changes. No trap: the failure is a mixed-up list rather
        # than one wrong value, and there is no single string to count.
        gold="Two were recorded wrongly: the mobile number and the unit's serial "
             "number. Everything else changed in the world rather than on paper - the "
             "plan, the delivery address, the billing address and the contact "
             "preference.",
        trap=None,
        kind="correction",
        # Both axes at once - the question is the split itself. Naming either one
        # would describe it as being about half of what it asks.
        closure=None,
    ),
    # --- unanswerable: an honest reader says it does not know ----------------------
    Question(
        id="q_pro_price",
        asked_at=_ASKED,
        text="What does the Pro plan cost per month on this account?",
        # Never stated. Pro is discussed at length on 3 March and 19 June and no price
        # is ever attached to it. The trap is the one price in the transcript, quoted
        # twice (9 April 13:58 and 6 August 19:47), which is for a single extra node
        # and not for a plan at all.
        gold="Not stated. The transcript never gives a price for the Pro plan.",
        trap="£79",
        kind="unanswerable",
        closure=None,
    ),
    Question(
        id="q_card_on_file",
        asked_at=_ASKED,
        text="Which card does this customer have on file?",
        # Never stated. A card is referred to on 5 August 9:38 and 6 August 19:40 and
        # is never identified - no issuer, no last four, no name. No trap: a system
        # that invents an answer here could invent anything, and the digit strings
        # floating around this corpus (a serial, a mobile, a postcode) are not one
        # candidate but several.
        gold="Not stated. The transcript mentions a card on file but never says which "
             "card, which bank, or any digits.",
        trap=None,
        kind="unanswerable",
        closure=None,
    ),
)


def conversation() -> list[Turn]:
    """The support history, oldest turn first.

    A fresh list of the same frozen turns on every call, so a caller that sorts or
    truncates it in place cannot change what the next caller sees.

    >>> turns = conversation()
    >>> turns is not conversation()
    True
    >>> [t.at for t in turns] == sorted(t.at for t in turns)
    True
    """
    return list(_CONVERSATION)


def questions() -> list[Question]:
    """The graded question set, in a stable order.

    >>> ids = [q.id for q in questions()]
    >>> len(ids) == len(set(ids))
    True
    >>> next(q for q in questions() if q.id == "q_plan_current").gold
    'Home.'
    >>> next(q for q in questions() if q.id == "q_plan_mid_march").gold
    'Pro.'
    """
    return list(_QUESTIONS)
