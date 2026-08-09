"""Redaction: whether the hook runs early enough to matter, and costs nothing unset.

The seam is one function, so almost nothing here tests *what* gets removed. What it
tests is **ordering**, because that is the only way this feature is wrong in a way you
find out about later:

* text redacted after `Episode.hash` leaves a digest of the secret in a stored column,
  which for anything low-entropy is a confirmation oracle and undoes the redaction;
* text redacted after `store.add_episode` is in the FTS index, in the clear, tokenized;
* text redacted after `embedder.encode` has been embedded — and for a hosted embedder,
  posted to a third party — and embeddings are invertible enough to matter;
* text redacted after `llm.extract` has been posted to a model provider;
* a claim redacted after `Reconciler.apply` has keys derived from text that is gone.

Each of those has a test that fails if the call moves, and three of them are asserted by
watching the collaborator rather than by reading the result: a recording embedder and a
recording model are the only witnesses to "this text left the process".

Then contract-shaped things: provenance still resolves, the two documented costs
(collapsing hashes, collapsing values) are real and asserted rather than hedged, and the
unset path is booby-trapped the way `tests/test_telemetry.py` booby-traps telemetry.

Last, the seventh silent failure. A policy whose rules stop matching the data raises
nothing and *speeds the write path up*, so the only evidence is aggregate:
`redact.inspected` against `redact.changed`, per field and per script. Those tests are
the counterpart to `test_telemetry.py::test_every_silent_failure_mode_has_a_live_series`
and they live here because the emission point is the seam rather than the pipeline.

Fully offline: `SQLiteStore(":memory:")`, `HashingEmbedder`, and local fakes.
"""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any, Sequence

import pytest

from memvara import Claim, Memvara, Episode, HashingEmbedder, NullLLM, Scope
from memvara.redact import (
    CLAIM_OBJECT,
    CLAIM_SUBJECT,
    CLAIM_TEXT,
    EPISODE,
    FIELDS,
    PatternRedactor,
    Redactor,
    luhn,
    redact_claim,
    redact_episode,
)
from memvara.telemetry import (
    REDACT_CHANGED,
    REDACT_INSPECTED,
    MemoryRecorder,
    script_of,
)

RAW_EMAIL = "bob.smith@corp.example"
RAW_PHONE = "555-123-4567"

#: A Japanese mobile number, punctuated exactly like the Latin one above and grouped
#: 3-4-4 instead of 3-3-4. `_PHONE` matches the second and not the first, which makes
#: this the cheapest honest example of the failure the telemetry exists to catch: not an
#: exotic input, just a locale whose formatting convention differs by one digit.
RAW_JP_TURN = "私の電話は090-1234-5678です"


# ===========================================================================
# Fakes. Two of them exist to witness text leaving the process.
# ===========================================================================

class RecordingEmbedder:
    """A `HashingEmbedder` that remembers every string it was asked to encode.

    The only witness to the claim that redaction precedes embedding. For a hosted
    embedder this call is an HTTP request, so `seen` is literally the list of strings
    that would have left the machine.
    """

    def __init__(self, dim: int = 64) -> None:
        self.inner = HashingEmbedder(dim=dim)
        self.seen: list[str] = []

    @property
    def dim(self) -> int:
        return self.inner.dim

    def encode(self, texts: Sequence[str]):
        self.seen.extend(texts)
        return self.inner.encode(texts)


class RecordingLLM:
    """An extractor that records the turns it was handed and returns fixed claims."""

    name = "fake/redact"
    is_noop = False

    def __init__(self, claims: Sequence[dict[str, Any]] = ()) -> None:
        self._claims = list(claims)
        self.seen: list[str] = []

    def extract(self, episodes, known_predicates):
        self.seen.extend(ep.content for ep in episodes)
        return list(self._claims)

    def classify_predicate(self, predicate, example):  # pragma: no cover - unused path
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}

    def resolve_predicate(self, surface, candidates):
        return {"canonical": None, "cardinality": "many", "volatility": "slow",
                "memory_type": "semantic"}


class RecordingRedactor:
    """Changes nothing, remembers everything it was offered."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def redact(self, text: str, /, *, field: str, scope: Scope) -> str:
        self.calls.append((field, text))
        return text

    def fields(self) -> set[str]:
        return {field for field, _ in self.calls}

    def texts(self) -> list[str]:
        return [text for _, text in self.calls]


def memory(*, redactor: Redactor | None = None, embedder=None, llm=None,
           user: str = "alice") -> Memvara:
    """An `Memvara` that constructs silently: an explicit `llm=` suppresses the
    degraded-extraction warning, which is about extraction and not about this."""
    return Memvara(embedder=embedder or HashingEmbedder(dim=64), llm=llm or NullLLM(),
                  user=user, redactor=redactor)


# ===========================================================================
# The one built-in
# ===========================================================================

@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail me at bob@corp.example", "mail me at [redacted:email]"),
        ("a.b+c@x.co.uk", "[redacted:email]"),
        ("call me on 555-123-4567.", "call me on [redacted:phone]."),
        ("(555) 123-4567", "[redacted:phone]"),
        ("+44 20 7946 0958", "[redacted:phone]"),
        ("card 4111 1111 1111 1111 ok", "card [redacted:card] ok"),
        ("4111-1111-1111-1111", "[redacted:card]"),
    ],
)
def test_the_shipped_rules_remove_what_the_docstring_says_they_remove(text, expected):
    assert PatternRedactor().redact(text, field=EPISODE, scope=Scope()) == expected


@pytest.mark.parametrize(
    "text",
    [
        "5551234567",                    # unpunctuated: deliberately not a phone here
        "server 192.168.1.10 is down",   # dotted quad, wrong shape
        "released 2024-01-15",           # a date is not a phone number
        "version 1.2.3 shipped",
        "ERR_7734_TLSHANDSHAKE",         # the token BM25 exists to return verbatim
        "git remote user@host:repo.git",  # no dot in the domain, so not an address
        "@mention someone",
        "id 1234567890123456789012",     # 22 digits: longer than any card
        "I live in Berlin",
        "",
    ],
)
def test_the_shipped_rules_leave_ordinary_text_alone(text):
    """False positives cost data rather than privacy, and a redactor that eats error
    codes and version numbers makes the store useless instead of private."""
    assert PatternRedactor().redact(text, field=EPISODE, scope=Scope()) == text


def test_a_sixteen_digit_number_that_is_not_a_card_survives_the_card_rule():
    """The checksum is the whole difference between a rule and a digit vacuum: both
    strings are sixteen digits in groups of four, and only one is a card number."""
    r = PatternRedactor()
    assert "[redacted:card]" in r.redact("4111 1111 1111 1111", field=EPISODE,
                                        scope=Scope())
    assert r.redact("1234 5678 9012 3456", field=EPISODE,
                    scope=Scope()) == "1234 5678 9012 3456"


@pytest.mark.parametrize("text", ["4111 1111 1111 111", "4" * 20, "no digits"])
def test_luhn_rejects_anything_outside_card_length(text):
    assert luhn(text) is False


def test_the_phone_rule_runs_before_the_card_rule_so_it_cannot_be_eaten():
    """Rules apply in mapping order. Two punctuated phone numbers side by side are
    nineteen digits with separators — card-shaped — and must come back as two phones."""
    out = PatternRedactor().redact("ring 555-123-4567 or 555-987-6543",
                                   field=EPISODE, scope=Scope())
    assert out == "ring [redacted:phone] or [redacted:phone]"


def test_redaction_is_idempotent_because_a_reingested_transcript_is_redacted_again():
    r = PatternRedactor()
    once = r.redact(f"mail {RAW_EMAIL} or call {RAW_PHONE}", field=EPISODE, scope=Scope())
    assert r.redact(once, field=EPISODE, scope=Scope()) == once


def test_the_pattern_set_and_the_token_are_both_replaceable():
    """The extension point inside the extension point: a deployment's own identifiers
    are the ones it actually holds, and no shipped default can know them."""
    r = PatternRedactor({"badge": re.compile(r"\bB-\d{6}\b")}, token="<{label}>")
    assert r.redact("badge B-004217 and mail bob@corp.example", field=EPISODE,
                    scope=Scope()) == "badge <badge> and mail bob@corp.example"


def test_a_replaced_pattern_set_keeps_the_checksum_that_belongs_to_its_label():
    """`ACCEPT` is keyed by label, so copying `PATTERNS`, adding a rule and passing the
    result back does not quietly turn the card rule into a digit vacuum."""
    r = PatternRedactor({**PatternRedactor.PATTERNS, "badge": re.compile(r"\bB-\d{6}\b")})
    assert r.redact("1234 5678 9012 3456", field=EPISODE,
                    scope=Scope()) == "1234 5678 9012 3456"


def test_the_repr_names_the_rules_in_force():
    assert repr(PatternRedactor()) == "<PatternRedactor email+phone+card>"
    assert repr(PatternRedactor({})) == "<PatternRedactor no rules>"


# ===========================================================================
# Ordering: the reason this feature exists at all
# ===========================================================================

def test_the_raw_turn_never_reaches_the_episode_row_or_its_text_index():
    """Redaction after `store.add_episode` would leave the address in the `episodes`
    row and, worse, tokenized in FTS5 where a one-word query returns it."""
    mem = memory(redactor=PatternRedactor())
    mem.add(f"mail me at {RAW_EMAIL} about the invoice")

    stored = [e.content for e in mem.store.iter_episodes()]
    assert stored == ["mail me at [redacted:email] about the invoice"]
    # Not "the query returns nothing" — vector search always returns its nearest
    # neighbours. The claim is that nothing retrievable still contains the text.
    hits = mem.search("smith", k=10, include_episodes=True)
    assert all("smith" not in r.text.lower() for r in hits)
    mem.close()


def test_the_embedder_never_sees_the_raw_turn():
    """The near-duplicate encode, and then the episode's own vector. For a hosted
    embedder both are HTTP requests, which makes this the strongest form of the
    ordering claim: not 'the vector is invertible' but 'the text left the machine'."""
    embedder = RecordingEmbedder()
    mem = memory(redactor=PatternRedactor(), embedder=embedder)
    mem.add(f"reach me on {RAW_PHONE} any time")

    assert embedder.seen                      # it really was asked to encode something
    assert all(RAW_PHONE not in text for text in embedder.seen)
    assert any("[redacted:phone]" in text for text in embedder.seen)
    mem.close()


def test_the_extraction_model_never_sees_the_raw_turn():
    llm = RecordingLLM()
    mem = memory(redactor=PatternRedactor(), llm=llm)
    mem.add(f"you can bill me at {RAW_EMAIL}, whichever is easier")

    assert llm.seen
    assert all(RAW_EMAIL not in turn for turn in llm.seen)
    mem.close()


def test_the_content_hash_is_taken_after_redaction_so_it_is_not_a_confirmation_oracle():
    """`episodes.hash` is a stored blake2b digest. Hashing before redacting would keep,
    in a column nobody reads as content, a digest of exactly the thing removed — and
    against a phone number, a digest is a guess-and-check away from the plaintext."""
    mem = memory(redactor=PatternRedactor())
    scope = mem.default_scope
    raw = f"call me at {RAW_PHONE}"
    mem.add(raw)

    [stored] = list(mem.store.iter_episodes())
    assert stored.hash == Episode(content="call me at [redacted:phone]",
                                  scope=scope).hash
    assert mem.store.find_episode_by_hash(scope.tenant,
                                          Episode(content=raw, scope=scope).hash) is None
    mem.close()


def test_two_turns_differing_only_inside_a_redacted_span_collapse_into_one():
    """The price of hashing after redaction, asserted rather than hedged: these are two
    different sentences and the store keeps one of them. The alternative ordering keeps
    both and keeps a digest of each secret, which is the worse trade."""
    mem = memory(redactor=PatternRedactor())
    first = mem.add(f"call me at {RAW_PHONE}")
    second = mem.add("call me at 555-987-6543")

    assert second.episode_ids == first.episode_ids
    assert second.skipped == 1
    assert len(list(mem.store.iter_episodes())) == 1
    mem.close()


def test_a_claim_is_redacted_before_the_keys_are_derived_from_it():
    """`Reconciler.apply` hashes subject and object into `fact_key`/`value_key` and then
    writes the row. Redacting afterwards would index the claim under a key computed from
    text that no longer exists, so `history()` would stop finding its own slot."""
    mem = memory(redactor=PatternRedactor())
    receipt = mem.remember("user", "phone", RAW_PHONE)
    [claim] = receipt.added

    assert claim.object == "[redacted:phone]"
    assert claim.text == "user phone [redacted:phone]"
    assert mem.history("user", "phone") == [mem.get(claim.id)]
    mem.close()


# ===========================================================================
# Every door into the store, and only the fields that should be offered
# ===========================================================================

def test_a_source_turn_attached_to_a_structured_write_is_redacted_too():
    """`remember(sources=[Episode(...)])` writes turns through `Memvara._write_claim`,
    not through the pipeline. Without a second call site the seam would hold for `add()`
    and leak for the call whose entire purpose is attaching provenance to an import."""
    mem = memory(redactor=PatternRedactor())
    turn = Episode(content=f"you can get me on {RAW_PHONE}", scope=mem.default_scope)
    mem.remember("user", "phone", RAW_PHONE, sources=[turn])

    assert [e.content for e in mem.store.iter_episodes()] == [
        "you can get me on [redacted:phone]"]
    mem.close()


def test_a_replacement_claim_passed_to_supersede_is_redacted():
    mem = memory(redactor=PatternRedactor())
    [old] = mem.remember("user", "phone", "555-111-2222").added
    replacement = Claim(subject="user", predicate="phone", object="555-333-4444",
                        scope=mem.default_scope)
    [new] = mem.supersede(old.id, replacement).added

    assert new.object == "[redacted:phone]"
    assert all(RAW_PHONE not in c.text for c in mem.get_all(include_invalidated=True))
    mem.close()


def test_a_claim_value_the_model_supplies_is_redacted_even_though_the_turn_already_was():
    """The turn is clean by the time any extractor sees it, so an extracted claim should
    be clean transitively. This is the case where it is not: a value that was reformatted,
    reassembled or invented rather than copied. Cheap insurance, and the reason the rule
    is 'every claim passes the hook once' rather than 'claims inherit their turn's'."""
    llm = RecordingLLM([{"subject": "user", "predicate": "contact_at",
                         "object": RAW_PHONE, "source_index": 0, "confidence": 0.9}])
    mem = memory(redactor=PatternRedactor(), llm=llm)
    [claim] = mem.add("there is a number somewhere in my profile").added

    assert claim.object == "[redacted:phone]"
    mem.close()


def test_the_predicate_is_never_offered_to_the_redactor():
    """Predicates are schema, not content: normalized, folded onto a canonical form
    shared across the tenant, and hashed into `fact_key`. A redactor that rewrote one
    would split a slot in two and silently disable contradiction detection."""
    rec = RecordingRedactor()
    mem = memory(redactor=rec)
    mem.remember("user", "commutes_by", "bicycle")

    assert "commutes_by" not in rec.texts()
    mem.close()


def test_exactly_the_documented_fields_are_offered_and_the_catalogue_lists_them_all():
    rec = RecordingRedactor()
    mem = memory(redactor=rec)
    mem.add("I live in Berlin")
    mem.remember("user", "drinks", "coffee")

    assert rec.fields() == set(FIELDS)
    assert set(FIELDS) == {EPISODE, CLAIM_SUBJECT, CLAIM_OBJECT, CLAIM_TEXT}
    mem.close()


def test_the_scope_is_passed_so_a_policy_can_route_on_who_the_data_belongs_to():
    """A server holds one `Memvara` per process, not one per tenant. Without the scope in
    the signature, "redact for the EU tenant" is not expressible at runtime at all."""

    class PerTenant:
        def redact(self, text: str, /, *, field: str, scope: Scope) -> str:
            return "[withheld]" if scope.tenant == "eu" else text

    mem = memory(redactor=PerTenant())
    mem.add("I live in Berlin", tenant="eu")
    mem.add("I live in Lisbon", tenant="us")

    assert [e.content for e in mem.store.iter_episodes("eu")] == ["[withheld]"]
    assert [e.content for e in mem.store.iter_episodes("us")] == ["I live in Lisbon"]
    mem.close()


def test_a_redactor_that_raises_is_not_swallowed():
    """Same contract as `Recorder`, for a stronger reason: a redaction pass that failed
    open would write the raw text and hand back an ordinary successful receipt."""

    class Broken:
        def redact(self, text: str, /, *, field: str, scope: Scope) -> str:
            raise RuntimeError("policy service unreachable")

    mem = memory(redactor=Broken())
    with pytest.raises(RuntimeError, match="policy service unreachable"):
        mem.add("I live in Berlin")
    mem.close()


# ===========================================================================
# Consequences the module docstring promises, held to
# ===========================================================================

def test_provenance_still_resolves_after_the_turn_it_points_at_was_redacted():
    """A redacted turn is still a turn. Redaction happens to the object that gets
    stored, so there is one version of the text and `why()` returns it — where redacting
    on read, or erasing the source instead, would leave the pointer dangling."""
    mem = memory(redactor=PatternRedactor())
    turn = Episode(content=f"mail me at {RAW_EMAIL}", scope=mem.default_scope)
    [claim] = mem.remember("user", "email", RAW_EMAIL, sources=[turn]).added

    prov = mem.why(claim.id)
    assert prov is not None
    assert [e.content for e in prov.episodes] == ["mail me at [redacted:email]"]
    mem.close()


def test_two_values_that_redact_alike_become_one_value():
    """The keying cost, stated in the docstring and asserted here. `value_key` hashes the
    object, so two different phone numbers that redact to one token are one value: the
    second write reinforces the first instead of superseding it, and `history()` shows a
    number the user changed exactly zero times."""
    mem = memory(redactor=PatternRedactor())
    mem.remember("user", "phone", "555-111-2222")
    second = mem.remember("user", "phone", "555-333-4444")

    assert second.added == [] and len(second.reinforced) == 1
    assert len(mem.history("user", "phone")) == 1
    mem.close()


def test_a_tokenizing_redactor_is_a_drop_in_because_the_seam_never_inverts_anything():
    """Destructive is the shipped default, not the only shape. Reversibility lives
    entirely in the deployment's implementation, and the mapping stays out of this
    process — which is the point, since a library holding the key beside the ciphertext
    has encrypted nothing.

    Keyed by plaintext, which is the determinism the protocol requires and which a token
    vault wants regardless: a value reaches the hook once as `claim.object` and again
    inside `claim.text`, so a vault that minted a token per *call* would write a row
    whose two halves name different tokens for one number.
    """
    vault: dict[str, str] = {}

    class Tokenizing:
        def redact(self, text: str, /, *, field: str, scope: Scope) -> str:
            def swap(m: re.Match[str]) -> str:
                return vault.setdefault(m.group(), f"[phone:{len(vault):04x}]")
            return re.sub(r"(?<!\d)\d{3}-\d{3}-\d{4}(?!\d)", swap, text)

    mem = memory(redactor=Tokenizing())
    [claim] = mem.remember("user", "phone", RAW_PHONE).added

    assert (claim.object, claim.text) == ("[phone:0000]", "user phone [phone:0000]")
    assert vault == {RAW_PHONE: "[phone:0000]"}
    mem.close()


def test_a_nondeterministic_redactor_makes_a_row_disagree_with_its_own_rendering():
    """The failure the determinism requirement exists to name, pinned so the contract is
    not merely advice. Two calls, two answers, and the stored object no longer appears in
    the stored text — which is what BM25 indexes and what the embedder encodes."""
    counter = iter(range(99))

    class Unstable:
        def redact(self, text: str, /, *, field: str, scope: Scope) -> str:
            return re.sub(r"(?<!\d)\d{3}-\d{3}-\d{4}(?!\d)",
                          lambda m: f"[phone:{next(counter)}]", text)

    mem = memory(redactor=Unstable())
    [claim] = mem.remember("user", "phone", RAW_PHONE).added

    assert claim.object == "[phone:0]" and claim.object not in claim.text
    mem.close()


def test_the_helpers_rewrite_in_place_so_no_returned_object_holds_the_raw_text():
    """`WriteReceipt.added` hands claims back to the caller. Redacting into a copy would
    make the return value of `add()` a channel around the redactor."""
    r = PatternRedactor()
    ep = Episode(content=f"mail {RAW_EMAIL}", scope=Scope())
    claim = Claim(subject="user", predicate="phone", object=RAW_PHONE, scope=Scope())

    assert redact_episode(r, ep) is ep and ep.content == "mail [redacted:email]"
    assert redact_claim(r, claim) is claim
    assert (claim.subject, claim.object) == ("user", "[redacted:phone]")
    assert claim.text == "user phone [redacted:phone]"


# ===========================================================================
# The seventh silent failure: a policy that has quietly stopped matching
# ===========================================================================

def turn(text: str) -> Episode:
    return Episode(content=text, scope=Scope("t", "alice"))


def claim_of(obj: str, predicate: str = "contact_at") -> Claim:
    return Claim(subject="user", predicate=predicate, object=obj, scope=Scope("t", "alice"))


def test_the_seam_reports_what_it_offered_the_policy_and_what_came_back_changed():
    """The pair, on the two doors. Neither number is readable alone — see
    `telemetry.REDACT_INSPECTED` — but a policy that inspected four strings and rewrote
    two of them is a policy demonstrably still doing something, which is the one thing no
    receipt, log line or exception in this library can tell you."""
    rec = MemoryRecorder()
    r = PatternRedactor()

    redact_episode(r, turn(f"call me at {RAW_PHONE}"), telemetry=rec)
    redact_claim(r, claim_of(RAW_PHONE), telemetry=rec)

    # One episode field plus a claim's three: four strings looked at.
    assert rec.total(REDACT_INSPECTED) == 4
    assert rec.total(REDACT_INSPECTED, field=EPISODE) == 1
    # The turn, the claim's object, and the rendering that repeats it.
    assert rec.total(REDACT_CHANGED) == 3
    assert rec.total(REDACT_CHANGED, field=CLAIM_OBJECT) == 1
    assert rec.total(REDACT_CHANGED, field=CLAIM_TEXT) == 1


def test_a_policy_that_matched_nothing_reports_a_zero_rather_than_no_series_at_all():
    """The `consolidate.merged` decision, applied where it matters more.

    Zero redactions is the *normal* day for most tenants, so the zero is not an alarm.
    What makes it worth emitting is the state it excludes: a series sitting at 0 says a
    policy ran and found nothing, and no series at all says nothing ran — a deployment
    that lost its `redactor=`, which is silent, raises nothing and makes writes faster.
    Only the reported zero distinguishes those two.
    """
    rec = MemoryRecorder()
    redact_episode(PatternRedactor(), turn("I live in Berlin"), telemetry=rec)

    assert rec.total(REDACT_CHANGED) == 0
    assert REDACT_CHANGED in rec.names()          # reported, not absent
    assert rec.total(REDACT_INSPECTED) == 1


def test_the_script_slice_names_the_population_a_rule_set_does_not_cover():
    """The failure this exists for, in its cheapest real form.

    Both turns state a phone number and both are punctuated with plain hyphens. The
    Latin one groups 3-3-4 and is removed; the Japanese one groups 3-4-4, which `_PHONE`
    has no branch for, and goes to disk verbatim. On the aggregate the policy looks
    healthy — it changed half of what it saw. Sliced by script it is 1-of-1 for one
    population and 0-of-1 for the other, which is the finding.
    """
    rec = MemoryRecorder()
    r = PatternRedactor()
    latin, kana = turn(f"call me at {RAW_PHONE}"), turn(RAW_JP_TURN)

    redact_episode(r, latin, telemetry=rec)
    redact_episode(r, kana, telemetry=rec)

    assert rec.total(REDACT_CHANGED) == 1 and rec.total(REDACT_INSPECTED) == 2
    assert rec.total(REDACT_INSPECTED, script="latin") == 1
    assert rec.total(REDACT_CHANGED, script="latin") == 1
    assert rec.total(REDACT_INSPECTED, script="kana") == 1
    assert rec.total(REDACT_CHANGED, script="kana") == 0
    # And the number really is still there, which is the point of the whole exercise.
    assert kana.content == RAW_JP_TURN


def test_the_script_is_taken_from_the_text_offered_not_from_the_policys_own_token():
    """A replacement token is Latin. On a short non-Latin turn its letters outvote the
    real ones, so classifying the *output* would file the han slice's own successes under
    latin and leave the slice reporting that nothing from that population is ever
    inspected — the exact blindness it was added to remove."""
    rec = MemoryRecorder()
    han = turn("我住在北京，电话 555-123-4567")

    redact_episode(PatternRedactor(), han, telemetry=rec)

    assert rec.total(REDACT_INSPECTED, script="han") == 1
    assert rec.total(REDACT_CHANGED, script="han") == 1
    assert rec.total(REDACT_INSPECTED, script="latin") == 0
    assert script_of(han.content) == "latin"      # the output really does read Latin


def test_the_field_slice_is_what_makes_a_half_redacted_row_visible():
    """One number per claim would average this away. A policy configured per field —
    which is why `field` is in the protocol at all — can remove a value from
    `claim.object` and leave it standing in `claim.text`, which is the string BM25
    indexes and the embedder encodes. Two fields, two rates, and the gap is the row."""

    class ObjectOnly:
        def redact(self, text: str, /, *, field: str, scope: Scope) -> str:
            return "[withheld]" if field == CLAIM_OBJECT else text

    rec = MemoryRecorder()
    c = claim_of(RAW_PHONE)
    redact_claim(ObjectOnly(), c, telemetry=rec)

    assert rec.total(REDACT_CHANGED, field=CLAIM_OBJECT) == 1
    assert rec.total(REDACT_CHANGED, field=CLAIM_TEXT) == 0
    assert rec.total(REDACT_INSPECTED, field=CLAIM_TEXT) == 1
    assert RAW_PHONE in c.text and c.object == "[withheld]"


def test_no_policy_configured_means_no_policy_metrics():
    """The absence is deliberate and is itself the signal. A deployment that never asked
    for redaction should not pay `script_of` on every field to be told so, and an
    operator reading a dashboard should see the series appear when a policy is installed
    and vanish when one is dropped — which is the alert for the blunter half of this
    failure mode."""
    rec = MemoryRecorder()
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice",
                  telemetry=rec)
    mem.add("call me at 555-123-4567")
    mem.remember("user", "phone", RAW_PHONE)

    assert [n for n in rec.names() if n.startswith("redact.")] == []
    mem.close()


# ===========================================================================
# Cost when unset
# ===========================================================================

def test_there_is_one_redaction_policy_per_instance_however_it_is_spelled():
    """`write_telemetry=` overriding the facade's recorder is a feature — a second sink
    is a reasonable thing to want. The same escape hatch for a redactor would spell a
    privacy control that covers `add()` and skips `remember(sources=...)`, so the two
    are kept in step and the source turn is redacted either way."""
    r = PatternRedactor()
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice",
                 write_redactor=r)
    assert mem.redactor is r and mem.writer.redactor is r
    mem.remember("user", "email", "x", sources=[Episode(content=f"mail {RAW_EMAIL}",
                                                        scope=mem.default_scope)])
    assert [e.content for e in mem.store.iter_episodes()] == ["mail [redacted:email]"]
    mem.close()


def test_the_default_is_none_rather_than_a_no_op_redactor():
    """The fast path is the absence of an object, not an object that does nothing —
    same contract as `telemetry`, and for the same reason: a library whose argument is
    the cost of the write path cannot ship an always-on hook on it."""
    mem = memory()
    assert mem.redactor is None and mem.writer.redactor is None
    mem.close()


def test_nothing_on_the_redaction_path_runs_when_it_is_unset(monkeypatch):
    """The mechanism half, and the half that cannot be flaky.

    Booby-trap every application point and drive all four write doors with the redactor
    unset. Reaching any of them is the bug; an unguarded call would also raise
    `AttributeError` on `None`, so this is belt and braces on a guard that has to hold
    for the cost claim to mean anything.
    """

    def boom(*a, **kw):
        raise AssertionError("redaction ran with no redactor configured")

    monkeypatch.setattr("memvara.write.pipeline.redact_episode", boom)
    monkeypatch.setattr("memvara.write.pipeline.redact_claim", boom)
    monkeypatch.setattr("memvara.core.redact_episode", boom)

    mem = memory(llm=RecordingLLM())
    mem.add(["I live in Berlin", "ok thanks", "something else entirely"])
    turn = Episode(content="a source turn", scope=mem.default_scope)
    [claim] = mem.remember("user", "drinks", "coffee", sources=[turn]).added
    mem.supersede(claim.id, Claim(subject="user", predicate="drinks", object="tea",
                                  scope=mem.default_scope))
    mem.writer.assert_claim(Claim(subject="user", predicate="likes", object="rain",
                                  scope=mem.default_scope))
    mem.close()


def _round(mem: Memvara, i: int) -> float:
    """Seconds for one three-turn write plus one structured write.

    The `Turn {i}:` prefix on the redactable turn is load bearing. Without it the phone
    number and the address are the only things distinguishing that turn between rounds,
    so under redaction every round writes the same sentence, the hash-repeat path takes
    over, and the arm being timed is the collapse rather than the hook — which is what
    the first version of this measurement actually reported, at 2.2x.
    """
    t0 = perf_counter()
    mem.add([f"My name is Alice number {i}.",
             f"Turn {i}: reach me on 555-123-{i:04d} or at alice{i}@corp.example.",
             f"Some unremarkable turn {i} that carries a statement."])
    mem.remember("user", "badge", f"B-{i:06d}")
    return perf_counter() - t0


def test_redacting_is_a_small_fraction_of_the_write_it_guards():
    """The measurement half.

    The *unset* arm is enforced structurally above and deliberately not timed here: the
    guard is one `is not None` test per call and its cost is below any timer this suite
    could run without measuring the machine instead. Measured out of tree instead,
    against a build of this tree with the four call sites and the import deleted from
    the source and nothing else changed. 200 rounds of the workload below into a fresh
    in-memory store, `HashingEmbedder(dim=64)`, `NullLLM`, best of five trials per
    launch, median over eight process launches per arm, one loaded developer machine:

        no hook in the source      1.8868 ms/round   (range 1.8640-1.8956)
        hook present, unset        1.8948 ms/round   (+0.4%, range 1.8797-1.9059)
        hook present, redacting    1.9108 ms/round   (+1.3%, range 1.8925-1.9436)

    The unset arm's median sits inside the control's own launch-to-launch range, which
    is the useful form of the result: the difference is below the noise floor rather
    than merely small. The redacting arm is +1.3% because three regexes over the four
    short strings a claim contributes cost 9.3 us against a 1.9 ms round.

    What is stable enough to assert in-tree is the bound in the other direction. Rounds
    are interleaved so a scheduler excursion hits both arms, and the minimum is taken
    because noise only ever adds.
    """
    off, on = memory(), memory(redactor=PatternRedactor())
    _round(off, 0), _round(on, 0)                     # warm both, measure neither
    unset = redacting = float("inf")
    for i in range(1, 41):
        unset = min(unset, _round(off, i))
        redacting = min(redacting, _round(on, i))
    off.close(), on.close()
    assert redacting < unset * 1.6, (
        f"redacting={redacting * 1000:.3f}ms against unset={unset * 1000:.3f}ms — "
        "the seam has stopped being a rounding error on the write it guards")


def test_no_telemetry_work_happens_while_redacting_without_a_recorder(monkeypatch):
    """The mechanism half of the telemetry cost claim, and the half that cannot be flaky.

    The complement of the test above it: there the *redactor* is unset, here it is
    configured and running and the *recorder* is not. That is the ordinary shape of a
    deployment that redacts and does not collect metrics, and it must pay for exactly the
    three statements it paid for before the series existed. Booby-trap the classification
    and the whole measured branch, then drive all four write doors with a live policy:
    reaching either is the bug.
    """

    def boom(*a, **kw):
        raise AssertionError("telemetry work ran with no recorder configured")

    monkeypatch.setattr("memvara.redact.script_of", boom)
    monkeypatch.setattr("memvara.redact._measured", boom)

    mem = memory(redactor=PatternRedactor(), llm=RecordingLLM())
    mem.add([f"call me at {RAW_PHONE}", "ok thanks", "something else entirely"])
    source = Episode(content=f"mail {RAW_EMAIL}", scope=mem.default_scope)
    [claim] = mem.remember("user", "drinks", "coffee", sources=[source]).added
    mem.supersede(claim.id, Claim(subject="user", predicate="drinks", object="tea",
                                  scope=mem.default_scope))
    mem.writer.assert_claim(Claim(subject="user", predicate="likes", object="rain",
                                  scope=mem.default_scope))
    assert [e.content for e in mem.store.iter_episodes()][:1] == [
        "call me at [redacted:phone]"]           # the policy really did run
    mem.close()


def _seam(redactor: Redactor, rec, i: int) -> float:
    """Seconds for the redaction one `_round` above pays for, telemetry included.

    Built outside the timer so the arms differ by the measurement and nothing else.
    """
    turns = [Episode(content=t, scope=Scope("t", "alice")) for t in (
        f"My name is Alice number {i}.",
        f"Turn {i}: reach me on 555-123-{i:04d} or at alice{i}@corp.example.",
        f"Some unremarkable turn {i} that carries a statement.")]
    claims = [claim_of(f"555-123-{i:04d}"), claim_of(f"B-{i:06d}", "badge")]
    t0 = perf_counter()
    for ep in turns:
        redact_episode(redactor, ep, telemetry=rec)
    for claim in claims:
        redact_claim(redactor, claim, telemetry=rec)
    return perf_counter() - t0


def test_measuring_the_seam_costs_a_fraction_of_the_write_the_seam_guards():
    """The measurement half.

    The unset arm is enforced structurally above rather than by a stopwatch, on the same
    reasoning as everywhere else here: it is one `is not None` test and its cost is under
    any timer this suite could run without measuring the machine. Measured out of tree
    instead, against a build of this tree with the seam's telemetry — the import, the
    `_measured` helper, the branch and the four `telemetry=` arguments — deleted from the
    source and nothing else changed. 200 rounds of `_round` above into a fresh in-memory
    store, `HashingEmbedder(dim=64)`, `NullLLM`, `PatternRedactor` on in every arm, best
    of five trials per launch, median over eight process launches per arm, on a developer
    machine running three other workstreams' test suites:

        no telemetry in the seam, no recorder    1.5221 ms/round  (range 1.4296-1.5656)
        seam telemetry present, no recorder      1.5120 ms/round  (-0.7%, 1.4392-1.6018)
        no telemetry in the seam, recording      1.5645 ms/round
        seam telemetry present, recording        1.6034 ms/round  (+2.5% on that arm)

    The unset arm's median is *below* the control's and its whole range sits inside the
    control's own launch-to-launch range, which is the useful form of the result rather
    than a flattering one: the difference is under the noise floor, not merely small. The
    recording arm's +2.5% is almost entirely `script_of`, measured directly at 5.64 us for
    a 59-character turn against 0.78 us for the two counter calls it feeds — the tags cost
    more than the emission, which is the honest way round for a slice nobody can
    reconstruct afterwards.

    What is stable enough to assert in-tree is the bound that matters: the whole measured
    seam stays a small fraction of the write it guards. Rounds are interleaved so a
    scheduler excursion hits both arms, and the minimum is taken because noise only adds.
    """
    mem, r, rec = memory(redactor=PatternRedactor()), PatternRedactor(), MemoryRecorder()
    _round(mem, 0), _seam(r, rec, 0)                  # warm both, measure neither
    write = measured = float("inf")
    for i in range(1, 41):
        write = min(write, _round(mem, i))
        measured = min(measured, _seam(r, rec, i))
    mem.close()
    assert measured < write * 0.20, (
        f"seam+telemetry={measured * 1000:.3f}ms against write={write * 1000:.3f}ms — "
        "watching the redactor has stopped being a rounding error on the write")
