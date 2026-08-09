"""The redaction seam: the last place text can be changed before it becomes durable.

Memvara already answers half of a privacy story properly. `erase()` and `purge()` remove
text that is *already* on disk — the claim row, the FTS entry that stores the tokens
directly, the embedding that leaks content under inversion, and optionally the source
turns. Most of the category cannot do that at all. This module is the other half, and
deliberately only the *seam* of it: one injectable hook that sees every string on its way
into the store and may hand back a different one.

One caveat on that first half, found while building this one and stated here because it
changes what this one is worth: neither call clears the `entities` table, whose
`canonical` column keeps the first spelling ever seen of every subject and every object.
After a full `purge()` the employers and cities are still there, verbatim. Redaction runs
*upstream* of entity resolution, so with a policy configured the surviving row reads
`[redacted:phone]` rather than the number — which makes this seam an accidental partial
mitigation for a bug it did not cause, and no substitute at all for fixing that bug.

**What is deliberately not here.** A PII ruleset worth the name, a compliance mode,
per-role policy, an audit report. Those are governance features; `docs/ROADMAP.md`
settles them as a closed product, and none of them belongs in an Apache-2.0 core. What
belongs here is the extension point and one honest default, for exactly the reason
`Recorder` is here and the dashboard that consumes it is not: a seam is worth nothing to
a competitor and everything to a deployment, and a library you have to fork in order to
comply is worse than one that ships no policy at all.

Where it runs, and why the order is the whole feature
-----------------------------------------------------

`WritePipeline.add` calls this before it does anything else with a turn. Everything that
happens to a turn's text afterwards, in order:

===================  ==========================================================
`ep.hash`            content-hash dedupe — and `episodes.hash` is a stored column
`store.add_episode`  the `episodes` row, and its FTS5 index entry
`embedder.encode`    a vector, and for a hosted embedder an HTTP request
`llm.extract`        the extraction model — another HTTP request
===================  ==========================================================

All four are downstream, and two of them leave the process. The usual argument for
redacting before the embedder is that embeddings are invertible enough to recover their
input, which is true and is why `erase()` deletes them; the blunter argument is that a
hosted embedder and a hosted extractor are third parties, and text that was posted to one
and redacted afterwards was never redacted.

The hash is the one that is easy to get backwards. Hashing first would be *tidier* — the
dedupe key would then be stable across a change of redaction policy, so tightening a rule
would not make old turns look new. It would also persist, in a column nobody thinks of as
content, a blake2b digest of precisely the text just removed. Against a low-entropy
secret — a phone number, a national ID, a card — a digest is a confirmation oracle:
anyone holding the file tests a guess in a microsecond, and the redaction is undone. So
the hash is taken **after** redaction, and the price is paid in the open: two turns
differing only inside a redacted span now hash identically, so the second is recorded as
an exact repeat. Its claims are reinforced rather than re-extracted and its own text is
never stored at all. Distinct-but-indistinguishable turns collapse. That is a real loss,
and it is the recoverable kind — the other ordering loses the secret.

The loss is not only informational, which is worth knowing before turning an aggressive
policy on. The exact-repeat path calls `WritePipeline._reinforcements_from_source`, which
scans every claim in the tenant, because the `Store` protocol carries no reverse
provenance index. That is affordable while repeats are rare, and redaction is precisely
what stops them being rare. Measured on an in-memory store with `HashingEmbedder`, a
workload whose only per-turn variation sat *inside* a redacted span cost 1.50, 1.99 and
3.02 ms per round at 100, 200 and 400 rounds, against 1.33, 1.45 and 1.69 for the same
workload left distinct: per-round cost rising with store size, so total cost quadratic.
Vary something outside the redacted span, or expect it.

What is offered, and what is not
--------------------------------

Four fields, listed in `FIELDS`: a turn's `content`, and a claim's `subject`, `object`
and `text`. Between them they are every string this library writes to a text column,
indexes for BM25, or hands to an embedder.

The **predicate is never offered**, and that is a decision rather than an oversight. A
predicate is schema, not content: it is normalized through `PredicateRegistry`, folded
onto a canonical form shared by every user in the tenant, and hashed into `fact_key`.
Rewriting it would split one slot into two and silently disable the contradiction
detection this library is built on, in exchange for censoring a controlled vocabulary
that no extraction path lets a user write into freely.

**`meta` is not offered either**, and that one is a gap rather than a decision. It is a
JSON column, so it does reach disk (it is neither indexed nor embedded). It is left alone
because it holds two incompatible things: whatever the caller passed as `**meta` or in a
transcript dict, which may well be personal, and machine fields — `salience_base`,
`last_observed_at`, `subject_entity` — where a rewrite corrupts ranking or keying
outright. A hook that cannot tell them apart would break the second to reach the first.
Structure your own metadata before you pass it.

Claims are redacted as well as turns, at both doors rather than once. Redacting the turn
first means the extractor never sees the raw text, so an *extracted* claim is clean by
construction — but `remember()`, `supersede()` and the mem0 importer write claims that
never had a turn, and those are exactly the calls an application uses when it already has
the data as structured fields. So every claim entering the store passes the hook exactly
once, whatever door it came through.

Consequences worth knowing before you turn it on
------------------------------------------------

* **Provenance still resolves.** Redaction happens to the object that gets stored, not as
  a filter over one already stored, so there is exactly one version of every turn and
  `why()` returns it. A redacted turn is still a turn; provenance that resolves to
  `"call me at [redacted:phone]"` is intact, and it is the only shape of this feature that
  keeps it so. Redacting on read, or erasing the source turn instead, dangles it.

* **Redaction changes keys.** `value_key` hashes the object and `fact_key` hashes the
  subject, both through the entity fold, so a redacted value is a *different value*. Two
  distinct phone numbers that redact to the same token become one value, and the second
  write reinforces the first rather than superseding it. A redactor that collapses
  *subjects* collapses slots, so two people redacted to one token contradict each other —
  which is why the shipped default matches nothing that appears in a subject position and
  why a redactor of your own should be tested against `history()`, not only against a
  string.

* **Nothing retroactive.** The hook applies to writes from the moment it is configured.
  Text already on disk stays there; `erase()` and `purge()` are the calls for that.

* **The default is destructive, and the seam does not require it to be.** `redact` returns
  text, and memvara never needs to invert it, so a tokenizing redactor that returns
  `"[phone:7f3a]"` and keeps the mapping in a vault the deployment controls is a drop-in
  with no change here. What memvara deliberately will not do is hold that mapping: a
  library storing the key beside the ciphertext is theatre, and the moment it did, the
  plaintext would be back inside this process and back inside `why()`.

Knowing it is still working
---------------------------

A redaction policy is the one part of this library whose failure is *rewarded*. Every
other seam that stops doing its job costs something visible — a slower query, a missing
answer, an exception. A `Redactor` whose rules stop matching the data raises nothing,
logs nothing, returns the string it was handed, and makes the write path measurably
faster; the store fills with the exact text the policy exists to remove, and the first
person to find out is an auditor. Nothing about that is hypothetical: the built-in rules
below match punctuated Latin-script phone numbers and a fixed email shape, so a tenant
switching to a new number format, a new locale, or a vendor id whose shape changed
crosses that line without anyone touching the code.

So the two helpers here take a `Recorder` and report `redact.inspected` and
`redact.changed`, tagged by field and by script. Neither number means much alone — see
`memvara.telemetry.REDACT_INSPECTED` for why the *ratio* is the signal, why the changed
count is emitted at zero, and why the absence of the pair is itself the alarm for the
blunter failure of a deployment that dropped its `redactor=`.

Both are reported by the seam rather than by any redactor, which is deliberate and is
what makes them worth having: a policy that self-reports can only report the misses it
knows about, and a policy's dangerous blind spots are by definition the ones it does not
know about. Measuring at the door instead means a deployment's own `Redactor` — the
serious case, and the one this module exists for — is measured on exactly the same terms
as the default, with nothing to implement.

Cost when unset
---------------

The default is `None`, not a no-op redactor, on the same terms as `telemetry`: an unset
redactor costs one `is not None` test per `add()` and per `assert_claim()` — not per turn,
not per claim, and no object is constructed, no list rebuilt and no string touched. A
library whose entire argument is the cost of the write path cannot ship an always-on hook
on it. See `tests/test_redact.py::test_nothing_on_the_redaction_path_runs_when_it_is_unset`.

The telemetry above is guarded a second time, independently, and everything it needs
computed — the script classification, the before/after comparison — lives inside that
guard rather than in front of it. A redactor with no recorder runs the same three
statements it ran before the series existed. See
`tests/test_redact.py::test_no_telemetry_work_happens_while_redacting_without_a_recorder`.

A redactor that raises propagates, and is not caught anywhere. That is the same contract
`Recorder` has and for a stronger reason: a redaction pass that failed open would write
the raw text and report success.
"""

from __future__ import annotations

import re
from typing import Callable, Mapping, Protocol

from .telemetry import REDACT_CHANGED, REDACT_INSPECTED, Recorder, script_of
from .types import Claim, Episode, Scope

__all__ = [
    "Redactor",
    "PatternRedactor",
    "redact_episode",
    "redact_claim",
    "luhn",
    "FIELDS",
    "EPISODE",
    "CLAIM_SUBJECT",
    "CLAIM_OBJECT",
    "CLAIM_TEXT",
]

#: A raw conversation turn, before it is hashed, stored, indexed or embedded. The
#: highest-value field by far: it is verbatim, unbounded, and the input to everything
#: downstream.
EPISODE = "episode"

#: A claim's subject. Almost always the literal `"user"`, because that is what extraction
#: emits — but `remember()` accepts anything, and a redactor that rewrites this is
#: rewriting a slot key. See the module docstring.
CLAIM_SUBJECT = "claim.subject"

#: A claim's object: the value of the fact, and the field a structured write puts a phone
#: number or an account number into.
CLAIM_OBJECT = "claim.object"

#: A claim's natural-language rendering — what gets embedded and BM25-indexed. Usually
#: the triple rendered by `Claim.render()`, and for `remember(text=...)` whatever the
#: caller supplied.
CLAIM_TEXT = "claim.text"

#: Every field a redactor is offered. Enumerable so a policy table, or a test that a new
#: call site was added to the documented set, can read it instead of grepping.
FIELDS = (EPISODE, CLAIM_SUBJECT, CLAIM_OBJECT, CLAIM_TEXT)


class Redactor(Protocol):
    """One method: text in, text out, before anything durable happens to it.

    Deliberately the smallest thing that can express a policy. `field` says which of
    `FIELDS` is being offered, so a deployment can be aggressive on raw turns and
    conservative on claim objects; `scope` carries tenant, user, agent and session, which
    is what makes "redact for EU tenants" expressible at all — a server holds one
    `Memvara` per process, not one per tenant, so without it there is no way to vary
    policy at runtime.

    Both are keyword-only and `text` is positional-only, so an implementation may call
    its own parameter whatever reads best. This is the whole contract and it is not
    expected to grow: anything needing more context than *which field, whose data* is a
    policy engine, and a policy engine is the closed governance layer's job.

    Implementations must be **pure with respect to memvara's state** — return a string,
    raise, or both; do not reach back into the store. They must be **deterministic**: the
    same input maps to the same output, every time and across processes. And they should
    be **idempotent**, because a re-ingested transcript is redacted again, and cheap,
    because this is on the write path.

    Determinism is the one of the three with teeth, and the failure is not obvious. A
    value that appears in a claim's `object` *and* in its rendered `text` is offered
    twice, so a tokenizing redactor that mints a fresh token per call writes a row whose
    text says `[phone:0001]` and whose object says `[phone:0000]` — and, across two
    writes, turns one phone number into two values that no longer dedupe. Key by the
    plaintext, which is what a token vault does anyway.

    They may hold state of their own: a tokenizing redactor keeping a mapping in a vault
    is the intended shape of a reversible policy, and is exactly why this is a protocol
    with a method rather than a bare callable.
    """

    def redact(self, text: str, /, *, field: str, scope: Scope) -> str:
        """Return `text` with whatever this policy removes from it removed."""
        ...


# --- applying it -------------------------------------------------------------
#
# Both helpers mutate in place and return the same object. That is not a shortcut: the
# receipt hands `added` claims and `episode_ids` straight back to the caller, and a
# redaction that produced a clean copy for the store while leaving the caller holding the
# original would make the return value of `add()` a channel around the redactor. One
# object, one version of the text, everywhere.
#
# Both also take an optional `telemetry`, and both branch on it rather than threading it
# through one code path. The unset branch is the same statement it was before the series
# existed - one attribute read, one call, one attribute write - which is the only way the
# cost claim above survives contact with a profiler.


def _measured(redactor: Redactor, text: str, field: str, scope: Scope,
              rec: Recorder) -> str:
    """Apply the policy to one string and report that it was looked at.

    Only ever reached with a recorder in hand, which is what keeps `script_of` and the
    comparison out of the unmeasured path entirely rather than merely out of an `if`.

    The comparison is the whole measurement: a `Redactor` reports nothing about its own
    work, so "did this string come back different" is the only evidence available at the
    seam, and it is available for every implementation rather than for the ones that
    chose to instrument themselves.
    """
    out = redactor.redact(text, field=field, scope=scope)
    # Classified on the input. The output is a mixture of the caller's text and the
    # policy's own replacement tokens, and bucketing a Han turn as Latin because
    # `[redacted:phone]` is Latin would corrupt precisely the slice that exists to show a
    # policy covering one population and no other.
    script = script_of(text)
    rec.counter(REDACT_INSPECTED, 1, field=field, script=script)
    # `0` rather than "no call": see `telemetry.REDACT_INSPECTED`. A policy that ran and
    # matched nothing and a policy that is not running are the two states an operator has
    # to tell apart, and only the reported zero does it.
    rec.counter(REDACT_CHANGED, 0 if out == text else 1, field=field, script=script)
    return out


def redact_episode(redactor: Redactor, episode: Episode, *,
                   telemetry: Recorder | None = None) -> Episode:
    """Redact a turn, in place, before anything derives from it.

    Must be called before `Episode.hash` is read, which is what `WritePipeline.add` does
    first. See the module docstring for why that ordering is not negotiable.
    """
    if telemetry is None:
        episode.content = redactor.redact(episode.content, field=EPISODE,
                                          scope=episode.scope)
    else:
        episode.content = _measured(redactor, episode.content, EPISODE, episode.scope,
                                    telemetry)
    return episode


def redact_claim(redactor: Redactor, claim: Claim, *,
                 telemetry: Recorder | None = None) -> Claim:
    """Redact a claim's three stored text fields, in place, before it is reconciled.

    Before reconciliation rather than after, because `Reconciler.apply` derives
    `fact_key`, `value_key` and the entity stamps from these strings and then writes the
    row. Redacting afterwards would leave keys computed over text that no longer exists.

    Three fields, three inspections. Counting the claim once instead would hide the thing
    the `field` slice is for: a value reaching the store through `claim.object` while the
    same policy leaves it standing in `claim.text` is a half-redacted row, and one
    per-claim number cannot say so.
    """
    scope = claim.scope
    if telemetry is None:
        claim.subject = redactor.redact(claim.subject, field=CLAIM_SUBJECT, scope=scope)
        claim.object = redactor.redact(claim.object, field=CLAIM_OBJECT, scope=scope)
        claim.text = redactor.redact(claim.text, field=CLAIM_TEXT, scope=scope)
    else:
        claim.subject = _measured(redactor, claim.subject, CLAIM_SUBJECT, scope, telemetry)
        claim.object = _measured(redactor, claim.object, CLAIM_OBJECT, scope, telemetry)
        claim.text = _measured(redactor, claim.text, CLAIM_TEXT, scope, telemetry)
    return claim


# --- the one built-in --------------------------------------------------------

#: Local part deliberately permissive, domain deliberately not: requiring at least one
#: dot and no consecutive separators is what stops `@mention` and `user@host` (a shell
#: prompt, a git remote) from matching.
_EMAIL = re.compile(r"\b[\w.%+'-]+@[\w-]+(?:\.[\w-]+)+\b")

#: Punctuated forms only. A bare `5551234567` is *not* matched, and that is the choice
#: that keeps this rule usable: ten unpunctuated digits are as likely to be an order
#: number, a timestamp or an account id, and a rule that eats those makes the store
#: useless rather than private.
_PHONE = re.compile(
    r"(?<!\d)(?:"
    r"\+\d{1,3}[ .\-]?\(?\d{1,4}\)?(?:[ .\-]?\d{2,4}){1,4}"   # +44 20 7946 0958
    r"|\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}"                      # (555) 123-4567
    r")(?!\d)"
)

#: A digit run of card length, with optional spaces or hyphens. On its own this matches
#: any long number; the `card` entry in `ACCEPT` is what makes it a rule rather than a
#: vacuum. See `luhn`.
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?!\d)")


def luhn(text: str) -> bool:
    """Whether the digits in `text` form a 13-19 digit Luhn-valid sequence.

    The reason the card rule is honest. A regex can only say "sixteen digits", which
    matches order numbers, tracking codes and concatenated timestamps; the checksum is
    what turns a 100% recall / near-0% precision pattern into one worth running. It
    rejects 9 in 10 non-card digit runs, which is a filter, not a guarantee.

    >>> luhn("4111 1111 1111 1111")
    True
    >>> luhn("1234 5678 9012 3456")
    False
    >>> luhn("42")
    False
    """
    digits = [int(c) for c in text if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _always(_match: str) -> bool:
    """Accept every match. The default for a pattern with no validator."""
    return True


class PatternRedactor:
    """Three regexes and a checksum. Useful, and not remotely compliance-grade.

    >>> from memvara.types import Scope
    >>> r = PatternRedactor()
    >>> r.redact("mail bob.smith@corp.example or call (555) 123-4567",
    ...          field=EPISODE, scope=Scope())
    'mail [redacted:email] or call [redacted:phone]'
    >>> r.redact("card 4111 1111 1111 1111, order 1234 5678 9012 3456",
    ...          field=CLAIM_TEXT, scope=Scope())
    'card [redacted:card], order 1234 5678 9012 3456'

    **What it will miss**, stated here rather than in a footnote, because a redactor
    whose limits are undocumented is worse than none — it converts a known exposure into
    a believed-safe one:

    * **Names, addresses, dates of birth, national identifiers, medical and financial
      detail in prose.** All of it. "I was diagnosed in March and my mother lives at 14
      Rue de la Paix" passes through untouched. There is no rule here for any of it,
      because there is no *pattern* for any of it — those need a model or a gazetteer,
      and both are the governance product.
    * **Unpunctuated digit runs.** `5551234567` is not a phone number as far as this is
      concerned. Deliberate: see `_PHONE`.
    * **Non-Latin scripts and spelled-out values.** "five five five, one two three" is
      nine words. So is the same number in Devanagari digits.
    * **Anything split across turns.** Each turn is redacted alone, so a number given as
      "my number is" / "555-123-4567" over two messages loses only the second half — and
      the first half is what makes the second findable.
    * **Everything not in `patterns`**: API keys and bearer tokens (`sk-…`, `AKIA…`,
      `ghp_…` — provider-specific, endless, and stale within a quarter), IBANs, SSNs and
      other national ID formats (jurisdictional), IP addresses, and postal codes. Add
      what your deployment actually holds; that is what the constructor is for.
    * **Semantics.** "the only left-handed cardiologist in Reykjavik" identifies exactly
      one person and contains no pattern at all.

    It also has false *positives*, which cost data rather than privacy: a hyphenated
    order number of the shape `100-200-3000` reads as a phone number, and roughly one in
    ten sixteen-digit references passes Luhn by chance. Both are visible in the stored
    text as a `[redacted:…]` token, which is the least bad way for a redactor to be
    wrong — the loss is legible instead of silent.

    **It does not report its own blind spots**, now that the seam around it can be
    measured, and that is a decision. Every entry in the list above is either something
    this class has no rule for (names, addresses, semantics) and therefore cannot count,
    or something it deliberately declines (`5551234567`) and could only count by running
    a second, shadow rule set — a detector that finds personal data and leaves it in
    place, which is the comprehensive ruleset `docs/ROADMAP.md` puts in the closed
    governance tier, carrying its cost on every write and none of its benefit. What an
    operator can act on is already emitted one level up: `redact.changed` against
    `redact.inspected`, sliced by `script`, shows a rule set matching a steady fraction
    of one population and nothing at all of another, which is the same finding, arrives
    for a redactor this class knows nothing about, and costs a comparison.

    Rules apply in mapping order, and the shipped order matters: emails first, then
    phones, then cards, so a punctuated phone number is gone before the card rule can
    consider a longer digit run that spans it. Replacement tokens contain no digits, so
    no rule can cascade into another's output.

    Pass `patterns` to replace the set entirely — `PatternRedactor({"badge":
    re.compile(r"\\bB-\\d{6}\\b")})` runs that and nothing else — and `token` to change
    the marker. A label present in `ACCEPT` gets its validator applied, which is how
    `card` gets its checksum; subclass to add another.
    """

    #: The shipped rules. Three, and stopping at three is the point: a fourth would be
    #: the first step toward the ruleset this module exists not to be.
    PATTERNS: Mapping[str, re.Pattern[str]] = {
        "email": _EMAIL,
        "phone": _PHONE,
        "card": _CARD,
    }

    #: Per-label validators, keyed by label so a caller who copies `PATTERNS`, adds an
    #: entry and passes the result back keeps the checksum on `card`.
    ACCEPT: Mapping[str, Callable[[str], bool]] = {"card": luhn}

    def __init__(self, patterns: Mapping[str, re.Pattern[str]] | None = None, *,
                 token: str = "[redacted:{label}]") -> None:
        self.patterns = dict(self.PATTERNS if patterns is None else patterns)
        self.token = token

    def redact(self, text: str, /, *, field: str, scope: Scope) -> str:
        """Substitute every accepted match. Ignores `field` and `scope`, by design.

        Both are in the signature because the protocol has them and a policy redactor
        needs them; this one applies the same three rules to every field of every scope,
        which is the only behaviour a built-in can honestly default to. Routing on
        tenant is a policy decision and belongs in the deployment's own implementation.
        """
        for label, pattern in self.patterns.items():
            token = self.token.format(label=label)
            accept = self.ACCEPT.get(label, _always)

            # Bound as defaults rather than closed over, so the callable `sub` invokes is
            # pinned to this iteration's label. Spelled as a `def` rather than the lambda
            # it used to be purely so the two parameters can carry annotations: mypy
            # cannot infer a lambda that has defaults, and the alternative was a
            # suppression on working code.
            def substitute(m: re.Match[str], replacement: str = token,
                           ok: Callable[[str], bool] = accept) -> str:
                return replacement if ok(m.group()) else m.group()

            text = pattern.sub(substitute, text)
        return text

    def __repr__(self) -> str:
        return f"<PatternRedactor {'+'.join(self.patterns) or 'no rules'}>"
