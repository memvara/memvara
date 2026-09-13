"""Distractor tickets that scale the support history without moving any fact.

    from demo.distractors import scale_conversation, scaled_conversation
    turns = scaled_conversation(10)        # 640 turns: the 64 authored plus 576 generated

The authored corpus in `demo/scenario.py` is sixty-four turns, and at that size a reader
handed the whole transcript gets every question right (`demo/README.md`, "What one run
produced"). The memory layer's argument there is a slope — retrieval context is flat in
corpus length while transcript context is linear — and one point does not measure a
slope. This module makes the second point: the same customer, the same product, the same
questions and golds, and a haystack N times longer.

## What a distractor may and may not say

Every generated turn belongs to a support ticket that could sit in this account's history
— a speed complaint, a firmware prompt, a guest network, an engineer's visit — and none of
them names the value of any fact a question is about, old or new. No address, no plan
name, no serial, no mobile number, no contact channel, no money. That constraint is what
keeps the trap metric meaningful at the larger size: the authored corpus states every
superseded value later and more often than the value that replaced it, and a distractor
that repeated either value would move that balance and change what a trapped answer
means. `tests/test_demo_scenario.py` checks the ban against a hand-written list of the
forbidden strings rather than against anything in this file.

A distractor may mention a fact's *topic* without its value — which subscription tiers
exist, where an invoice appears, whether somebody has to be in when a parcel comes — and
several do, deliberately. Topic words are what a retriever matches on, so those tickets
compete with the authored turns for the twelve slots a retrieval arm has, which is the
pressure the second size exists to apply. The two controls (the account name and the
billing day) and the two facts the corpus leaves unstated (the Pro plan's price and the
card on file) are kept out as well, so the unanswerable questions stay unanswerable and
the controls stay stated exactly once.

## How the turns are made and placed

Whole tickets, in a fixed cycle over `TICKETS`, each filled in with details that depend on
the ticket's index and date — a reference, a firmware version, a room, a speed reading —
so that every generated turn is unique text. That uniqueness is load-bearing rather than
cosmetic: `Memvara.add()` returns the existing episode for a repeat whose role and text
hash the same, so a repeated turn would leave the two memvara arms holding fewer turns
than `full_transcript`, and the arms would no longer be reading the same haystack.
`scale_conversation` refuses to return a corpus with a repeat in it.

Tickets land on days that carry no authored turn, at 06:00 or 22:00, three minutes
between turns. So a generated turn never shares an instant with an authored one, and a
generated ticket is never within six hours of an authored one, which is the gap the
corpus's shape test uses to tell tickets apart — the scaled corpus therefore keeps the
shape the authored one is tested for: tickets opened by the customer, speakers alternating
within a ticket. Everything stays strictly inside the authored window, so `asked_at`
cutoffs slice the generated turns exactly as they slice the authored ones. No clock, no
randomness, no I/O: `scaled_conversation(10) == scaled_conversation(10)`, always.

`scale_conversation(turns, factor)` is the general form and works on any dated
conversation of frozen `Turn`-shaped rows; `scaled_conversation(factor)` applies it to
the authored one. Factor 1 returns the input untouched, which is why `--corpus-scale 1`
is the default on `demo/harness.py` and the offline run's report is byte-identical to
what it was before this module existed.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta
from typing import Any, Sequence

from demo.scenario import Turn, conversation

__all__ = ["TICKETS", "scale_conversation", "scaled_conversation"]


#: The ticket templates: (role, text) pairs, the customer first and the speakers
#: alternating. Every text carries `{ref}` or `{date}`, the two slots that are unique to
#: a ticket, which is what makes every generated turn unique — see the module docstring.
#: British support-desk register, the same customer and product as the authored corpus,
#: and not one value of a fact a question is about.
TICKETS: tuple[tuple[tuple[str, str], ...], ...] = (
    # A speed complaint from one part of the workshop.
    (
        ("user", "Ticket {ref}. The link in {room} has been crawling since {day} {date}: "
                 "a speed test on the workshop laptop just gave me {down} down and {up} "
                 "up, and it was double that last month."),
        ("assistant", "Thanks for the figures on {ref}. {down} down is well under what "
                      "the node in {room} should deliver. Can you run the same test "
                      "standing next to the main unit, so we can tell the node from the "
                      "line?"),
        ("user", "Next to the main unit it's fine, {down2} down and {up2} up. It's only "
                 "{room} that suffers, and that's where {device} lives; it has been that "
                 "way since {date}."),
        ("assistant", "That points at the node in {room} rather than the line. {device} "
                      "is most likely holding that node on the slower band, so I've "
                      "pushed a band-steering change for {ref} that takes effect on the "
                      "node's next restart."),
        ("user", "Restarted it on {day} evening and {room} is back to normal. Thanks, "
                 "that's {ref} sorted."),
        ("assistant", "Glad to hear it. {ref} is closed; if {room} drops again, message "
                      "us with the date and we'll reopen it."),
    ),
    # A firmware prompt.
    (
        ("user", "The app is nagging me about firmware {fw} for the main unit, {ref} on "
                 "your side apparently. Is it safe to run during the working day or will "
                 "it knock everything off?"),
        ("assistant", "{fw} is safe to install whenever suits, {ref} noted. Expect about "
                      "four minutes of downtime while the main unit restarts; the nodes "
                      "follow one at a time after it."),
        ("user", "Ran it at lunch on {day} {date}. Everything came back, and {device} "
                 "reconnected on its own."),
        ("assistant", "Good: the update on {ref} is showing as complete here, and {fw} is "
                      "now on every node."),
    ),
    # A guest network for the apprentices.
    (
        ("user", "I've got {n} apprentices starting on {day} {date}, {ref} if you want a "
                 "reference. Can I give them Wi-Fi without giving them the workshop "
                 "network?"),
        ("assistant", "Yes: set up a guest network under Networks in the app, {ref} "
                      "noted. It gets its own name and passphrase and can't see anything "
                      "on the main network."),
        ("user", "Done, and named it after the workshop. Can I cap what the {n} of them "
                 "use on it? Reference {ref}."),
        ("assistant", "You can. The guest network has its own speed limit under Advanced; "
                      "a limit of {cap} Mbps each leaves {device} unaffected. Noted on "
                      "{ref}."),
        ("user", "Set to {cap} on {date}. That's all sorted."),
        ("assistant", "Closing {ref} then. The guest passphrase can be rotated from the "
                      "same screen whenever the {n} of them move on."),
    ),
    # An engineer's visit for a node that keeps rebooting.
    (
        ("user", "The node in {room} keeps rebooting itself, roughly hourly since {date}. "
                 "Reference {ref}. I'd rather someone looked at it than swapped it "
                 "blind."),
        ("assistant", "Understood; I've booked an engineer for {ref}. The first slot is "
                      "{day} between {hour} and {hour_end}. Does that work for the "
                      "workshop?"),
        ("user", "{day} is bad, I'm out fitting a staircase. Can they come the day after, "
                 "same hours? Still {ref}."),
        ("assistant", "Moved to the day after, {hour} to {hour_end}, and the engineer "
                      "will message you when they're twenty minutes away. Does anyone "
                      "need to be on site to open up {room} for {ref}?"),
        ("user", "No, {room} is open during the day. I'll leave the node where it is "
                 "until they come. Reference {ref}."),
        ("assistant", "Booked and confirmed on {ref}. The visit report will appear in the "
                      "app the same evening."),
    ),
    # A content filter on one device.
    (
        ("user", "Is there a way to block video sites on {device} only? Reference {ref}. "
                 "It's meant for orders, not for whoever's on tea break."),
        ("assistant", "Yes: profiles in the app, {ref}. Put {device} in its own profile "
                      "and switch on the video filter for that profile alone; nothing "
                      "else in the workshop is affected."),
        ("user", "Profile made on {date}, filter on. It's working, and nothing else "
                 "noticed."),
        ("assistant", "Thanks for confirming, {ref} closed. The profile also lets you "
                      "pause that device's access at a fixed time each day if you ever "
                      "want to."),
    ),
    # A maintenance notice.
    (
        ("user", "I've had a notice about network maintenance in the early hours of {day} "
                 "{date}, {ref}. Will {device} lose its connection for long?"),
        ("assistant", "The {ref} maintenance window is up to forty minutes between two "
                      "and three in the morning. {device} reconnects on its own once the "
                      "line is back; nothing in the app needs touching."),
        ("user", "Fine, nobody's in {room} at that hour anyway. Thanks for the warning on "
                 "{ref}."),
        ("assistant", "Noted on {ref}. If anything is still offline at nine that morning, "
                      "reopen this ticket and we'll look straight away."),
    ),
    # Locked out of the app.
    (
        ("user", "Locked out of the app since {day} {date}: it says my passcode is wrong "
                 "every time. {ref} on the notification."),
        ("assistant", "Sorry about that, {ref}. Tap 'Forgotten passcode' on the sign-in "
                      "screen; a six-digit code shows on the main unit's display for two "
                      "minutes, and entering it lets you set a new one."),
        ("user", "Got the code off the display on {date}, new passcode set, and I'm back "
                 "in."),
        ("assistant", "Closing {ref}. The display code is the only recovery route, so "
                      "it's worth keeping the main unit somewhere you can see it."),
    ),
    # Interference from a neighbouring network.
    (
        ("user", "Since the unit next door opened on {date}, the Wi-Fi in {room} drops "
                 "for a few seconds every couple of minutes. Reference {ref}."),
        ("assistant", "That pattern on {ref} is usually a neighbouring network on the "
                      "same channel. The Network Health page under Advanced in the app "
                      "shows the channels in use around you; what does it list for "
                      "{room}?"),
        ("user", "It shows channel {chan} for us and channel {chan} for two other "
                 "networks, and it has been like that since {date}."),
        ("assistant", "That's the clash. I've moved your node in {room} to channel "
                      "{chan2} for {ref}; the change applies within a minute and no "
                      "device needs reconnecting."),
        ("user", "Solid since {day} evening. Thanks, that's {ref} done."),
        ("assistant", "Good, {ref} closed. The app will now pick a channel itself if the "
                      "neighbours change again."),
    ),
    # A device that only speaks the slower band.
    (
        ("user", "Trying to get {device} onto the network and it just won't see it, "
                 "{ref}. It joined my old router fine."),
        ("assistant", "Most of those only speak the slower 2.4 GHz band, {ref}. In the "
                      "app, open the network settings and switch on 'show bands "
                      "separately' for ten minutes; {device} will then see a network it "
                      "can join."),
        ("user", "That did it, joined on {date}. Do I turn the setting back off?"),
        ("assistant", "Yes: once it's joined it stays joined, and merging the bands again "
                      "keeps everything else fast. Closing {ref}."),
    ),
    # Data usage, and where the subscription's allowance is shown (the topic, no value).
    (
        ("user", "The workshop cameras feel like they're eating data, {ref}. Where do I "
                 "see what the subscription allows and what the cameras are using?"),
        ("assistant", "Under Subscription in the app, {ref}: the allowance for your "
                      "current tier is at the top and the usage by device is below it. "
                      "The cameras will be near the top of that list."),
        ("user", "Found it on {date}. They're using about {gb} GB a month, well inside "
                 "the allowance."),
        ("assistant", "Then nothing needs changing. Closing {ref}; the same page shows "
                      "the other tiers if you ever want to compare them."),
    ),
    # Where an invoice appears (the topic, not where paper ones are sent).
    (
        ("user", "Where do I find last month's invoice, {ref}? I need it for the "
                 "accountant by {day}."),
        ("assistant", "Invoices are under Billing in the app as soon as they're issued, "
                      "{ref}, and each one can be downloaded as a PDF from there."),
        ("user", "Downloaded on {date}, thanks. Does the app keep the older ones too?"),
        ("assistant", "Every invoice since the account opened is on that page. Closing "
                      "{ref}."),
    ),
    # A parcel on its way (the topic of delivery, not the address).
    (
        ("user", "There's a replacement power lead on its way to me, {ref}. Do I need to "
                 "be at the property when it comes, or can the courier leave it?"),
        ("assistant", "The courier will use the safe place you've set in the app for "
                      "{ref}, so nobody needs to be there. It's due {day}."),
        ("user", "Set the safe place to the timber store porch on {date}. Reference "
                 "{ref}."),
        ("assistant", "Updated on {ref}; the courier sees that the moment it's saved."),
    ),
)

#: The details a template is filled in with. `ref` and `date` are unique to a ticket; the
#: rest cycle over these, so the prose varies rather than repeats.
ROOMS = ("the paint room", "the machine shop", "the timber store", "the office",
         "the spray booth", "the loading bay", "the mezzanine", "the finishing room")
DEVICES = ("the label printer", "the CNC controller", "the dust-extractor timer",
           "the office tablet", "the kiln sensor", "the booking tablet",
           "the workshop speaker", "the thermostat")
NUMBERS = ("two", "three", "four", "five")
HOURS = (("nine", "eleven"), ("ten", "twelve"), ("eleven", "one"), ("one", "three"),
         ("two", "four"))
CHANNELS = ((1, 6), (6, 11), (11, 1))

#: Tickets open at one of these hours on a day with no authored turn. Neither hour is one
#: the authored corpus uses, and the two are sixteen hours apart, so two tickets on one
#: day are still two tickets under the six-hour rule.
OPENING_HOURS = (6, 22)
MINUTES_BETWEEN_TURNS = 3


def _details(index: int, when: datetime) -> dict[str, Any]:
    down, up = 38 + (index * 7) % 60, 9 + (index * 3) % 14
    start, end = HOURS[index % len(HOURS)]
    channel, other = CHANNELS[index % len(CHANNELS)]
    return {
        "ref": f"WR-{1200 + 7 * index}",
        "date": f"{when.day} {when:%B}",
        "day": f"{when:%A}",
        "room": ROOMS[index % len(ROOMS)],
        "device": DEVICES[(index * 3) % len(DEVICES)],
        "fw": f"4.{2 + index % 6}.{index % 9}",
        "down": down, "up": up, "down2": down + 40 + index % 5, "up2": up + 6,
        "n": NUMBERS[index % len(NUMBERS)],
        "cap": 20 + 5 * (index % 3),
        "gb": 40 + (index * 11) % 90,
        "chan": channel, "chan2": other,
        "hour": start, "hour_end": end,
    }


def _sentence(text: str) -> str:
    """A template that opens with a slot gets a capital where a sentence starts."""
    return text[0].upper() + text[1:]


def scale_conversation(turns: Sequence[Turn], factor: int) -> list[Turn]:
    """`turns` padded to `factor` times its length with generated tickets, in order.

    Factor 1 is the input itself, as a list. Above that, whole tickets are generated in a
    fixed cycle over `TICKETS` and spread evenly over the openings the input leaves free,
    then cut to exactly the count needed; the last ticket may therefore end mid-exchange,
    which real support histories also do. Refuses a factor below 1, a conversation too
    dense to leave enough free days for the tickets it would need, and — as a guard on the
    templates rather than on the caller — any generated turn whose role and text repeat.
    """
    if factor < 1:
        raise ValueError(f"factor must be at least 1, got {factor}")
    authored = sorted(turns, key=lambda t: t.at)
    if factor == 1 or not authored:
        return list(turns)
    wanted = (factor - 1) * len(authored)

    busy = {t.at.date() for t in authored}
    first, last = authored[0].at.date(), authored[-1].at.date()
    tzinfo = authored[0].at.tzinfo
    openings = [
        datetime.combine(first + timedelta(days=offset), time(hour), tzinfo=tzinfo)
        for offset in range(1, (last - first).days)
        if first + timedelta(days=offset) not in busy
        for hour in OPENING_HOURS
    ]

    # How many whole tickets it takes to reach `wanted`, so they can be spread evenly.
    tickets, planned = 0, 0
    while planned < wanted:
        planned += len(TICKETS[tickets % len(TICKETS)])
        tickets += 1
    if tickets > len(openings):
        raise ValueError(
            f"scaling by {factor} needs {tickets} tickets and the conversation leaves "
            f"only {len(openings)} free openings between {first} and {last}")

    generated: list[Turn] = []
    for index in range(tickets):
        opened = openings[index * len(openings) // tickets]
        details = _details(index, opened)
        for position, (role, text) in enumerate(TICKETS[index % len(TICKETS)]):
            generated.append(replace(
                authored[0], role=role,
                at=opened + timedelta(minutes=position * MINUTES_BETWEEN_TURNS),
                text=_sentence(text.format(**details)),
            ))
    generated = generated[:wanted]

    seen = {(t.role, t.text) for t in authored}
    for turn in generated:
        if (turn.role, turn.text) in seen:
            raise ValueError(f"generated turn repeats an earlier one: {turn.text!r}")
        seen.add((turn.role, turn.text))
    return sorted([*turns, *generated], key=lambda t: t.at)


def scaled_conversation(factor: int) -> list[Turn]:
    """The authored corpus, scaled. `scaled_conversation(1) == conversation()`."""
    return scale_conversation(conversation(), factor)
