"""Confirm tokens for the bulk closures (`memory_end_matching`, `memory_forget_matching`).

The token is what makes "close everything that matches this query" safe: the first call
returns the matches and a signed token and changes nothing; the second passes the token
back and closes exactly the ids the first call listed. A token an attacker can forge,
replay after it expires, reuse for the other closure, or carry into another scope would
turn that safety into a way to close memories the caller never saw. Each of those is
refused, and nothing is applied. See `memvara/confirm.py`.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timedelta, timezone

import pytest

from harness import stores
from memvara.confirm import CONFIRM_TTL, ConfirmationRefused, Confirmer

#: A fixed instant so the tests never depend on the wall clock. The token binds an expiry
#: `CONFIRM_TTL` after this, and every `check` below passes its own `now`.
NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)

#: A shared secret, so a second `Confirmer` built with it stands in for a second worker
#: process serving one store. The per-process default key is tested separately.
SECRET = "a shared confirmation secret"


def _issued() -> "tuple[Confirmer, str]":
    """A confirmer and a token it issued to end two claims."""
    confirmer = Confirmer(SECRET)
    token, _expires = confirmer.issue(["cl_b", "cl_a"], "ended", now=NOW)
    return confirmer, token


def test_a_valid_token_round_trips_and_returns_its_sorted_ids() -> None:
    """The honest path: a token this key issued, presented before it expires for the
    closure it was issued for, returns exactly the ids it bound, sorted."""
    confirmer, token = _issued()
    assert confirmer.check(token, "ended", now=NOW) == ["cl_a", "cl_b"]


@pytest.mark.parametrize("forge", [
    lambda tok: tok[:-1] + ("0" if tok[-1] != "0" else "1"),  # a flipped mac character
    lambda tok: tok.split(".")[0] + ".deadbeef",              # a replaced mac
    lambda tok: "not-a-token",                                # no separator at all
    lambda tok: "",                                           # empty
    lambda tok: tok + "trailing",                             # bytes appended to the mac
])
def test_a_forged_or_altered_token_is_refused(forge) -> None:
    """A token whose body or mac does not verify under this key is refused before any
    field inside it is parsed, so a forgery never reaches `json.loads`."""
    confirmer, token = _issued()
    with pytest.raises(ConfirmationRefused):
        confirmer.check(forge(token), "ended", now=NOW)


def test_a_token_replayed_after_it_expires_is_refused() -> None:
    """A token found later — in a log, a transcript — confirms nothing once `CONFIRM_TTL`
    has passed since the preview."""
    confirmer, token = _issued()
    later = NOW + CONFIRM_TTL + timedelta(seconds=1)
    with pytest.raises(ConfirmationRefused):
        confirmer.check(token, "ended", now=later)


def test_a_token_issued_for_one_closure_is_refused_for_the_other() -> None:
    """A preview confirms one closure only. A token issued to end memories cannot be used
    to retire them, because the two are opposite claims about the past."""
    confirmer, token = _issued()
    with pytest.raises(ConfirmationRefused):
        confirmer.check(token, "retired", now=NOW)


def test_a_token_from_another_key_is_refused() -> None:
    """A token this store's key did not issue is refused, so a token minted by any other
    process or by an attacker with a different secret cannot close anything here."""
    _confirmer, token = _issued()
    with pytest.raises(ConfirmationRefused):
        Confirmer("a different secret").check(token, "ended", now=NOW)


def test_a_non_ascii_token_is_refused_cleanly_rather_than_raising() -> None:
    """A token whose body is not valid base64 — a non-ASCII string — is refused as a
    `ConfirmationRefused`, not some other error. The refusal happens at the mac check, so
    the payload is never decoded and no unexpected exception reaches the caller."""
    confirmer, _token = _issued()
    with pytest.raises(ConfirmationRefused):
        confirmer.check("токен.deadbeef", "ended", now=NOW)


def test_a_token_cannot_be_confirmed_from_another_scope(tmp_path: pathlib.Path) -> None:
    """Two handles on one store file share a confirmation secret, so each can verify the
    other's tokens — and it still does not let one scope close another's memories. A
    preview taken as `alice` lists a claim only `alice` can see; confirming it as `bob`
    is refused, because the claim is not visible to `bob`, even though the mac verifies.
    """
    db = tmp_path / "store.db"
    with stores.file(db, confirm_secret=SECRET, user="alice") as alice:
        alice.remember("user", "works_at", "AcmeSecret")
        preview = alice.forget_matching("AcmeSecret", close="ended", k=5)
        assert list(preview.matches.values()) == ["user works at AcmeSecret"]
    with stores.file(db, confirm_secret=SECRET, user="bob") as bob:
        with pytest.raises(ConfirmationRefused):
            bob.forget_matching("anything", close="ended", k=5, confirm=preview.confirm)
