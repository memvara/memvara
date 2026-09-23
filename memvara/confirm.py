"""The confirmation token behind `Memvara.forget_matching`.

Closing every memory that matches a query is two calls. The first returns the matches and
a token and changes nothing. The second passes the token back and closes exactly the ids
the first call listed. This module issues and checks that token.

**What the token binds.** It is an HMAC-SHA256 over the sorted claim ids, the closure
(`ended` or `retired`) and an expiry ten minutes after the preview. The ids travel inside
the token, so the confirming call does not search again: it closes the list the caller
saw, and a claim that started matching the query in between is not swept up with it.

**Where the key comes from.** A key given to the constructor (`MEMVARA_CONFIRM_SECRET` on
a server) is used as it is, so a preview served by one worker process can be confirmed
by another that holds the same key. With no key configured, the library generates one
per process, at import. That is correct for a single local process, and it means a token does not
survive a restart, which is the safe direction for a token whose job is to prove the
caller saw what is about to change.

**What a refusal means.** `ConfirmationRefused` is raised for a token this key did not
issue or that was altered, for one past its expiry, and for one issued for the other
closure. `Memvara.forget_matching` raises it too when a listed claim is no longer live
or no longer visible. In every case nothing has been applied.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Sequence

from .types import Closure, as_utc

__all__ = ["CONFIRM_TTL", "ConfirmationRefused", "Confirmer"]

#: How long a preview's token is accepted. Long enough to read a list of twenty matches
#: and decide; short enough that a token found later in a log confirms nothing.
CONFIRM_TTL = timedelta(minutes=10)


class ConfirmationRefused(ValueError):
    """A confirmation token was refused, and nothing was applied.

    The message says which of the reasons it was, so a caller can tell a token that
    expired (preview again) from one whose claims changed (read the new preview before
    deciding again).
    """


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


#: The key used when none is configured. Generated once per process, at import, so every
#: `Memvara` in one process accepts every other's tokens and no other process does.
_PROCESS_KEY = secrets.token_bytes(32)


class Confirmer:
    """Issues and checks confirmation tokens under one key.

    `secret` is the configured key, as text or bytes, or `None` for this process's own
    generated key. Two confirmers built with the same secret accept each other's tokens,
    which is what lets a preview and its confirmation land on different processes.

    >>> c = Confirmer("a shared secret")
    >>> now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    >>> token, expires = c.issue(["cl_b", "cl_a"], "ended", now=now)
    >>> c.check(token, "ended", now=now)
    ['cl_a', 'cl_b']
    >>> Confirmer("a shared secret").check(token, "ended", now=now)
    ['cl_a', 'cl_b']
    >>> c.check(token, "retired", now=now)            # doctest: +ELLIPSIS
    Traceback (most recent call last):
    memvara.confirm.ConfirmationRefused: this token was issued to end memories...
    """

    __slots__ = ("_key",)

    def __init__(self, secret: str | bytes | None = None) -> None:
        if secret is None:
            key = _PROCESS_KEY
        elif isinstance(secret, str):
            key = secret.encode("utf-8")
        else:
            key = bytes(secret)
        if not key:
            # An empty key is a valid HMAC key, and every process configured with it
            # would accept every other process's tokens and anybody's forgeries. A blank
            # setting is a mistake, so it is refused rather than taken literally.
            raise ValueError(
                "the confirmation secret is empty. Leave it unset to generate one per "
                "process, or set the same non-empty value on every process that serves "
                "this store.")
        self._key = key

    def _mac(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def issue(self, ids: Sequence[str], close: Closure, *,
              now: datetime) -> tuple[str, datetime]:
        """A token binding `ids`, `close` and an expiry `CONFIRM_TTL` after `now`.

        Returns the token and the instant it expires. The ids are sorted before they are
        bound, so the order the search ranked them in does not change the token.
        """
        # Whole seconds, and the instant returned is the instant encoded. Returning the
        # untruncated one told a caller the token lived up to a second longer than
        # `check` would honour it.
        stamp = int((as_utc(now) + CONFIRM_TTL).timestamp())
        expires = datetime.fromtimestamp(stamp, tz=timezone.utc)
        payload = json.dumps(
            {"close": close, "expires": stamp, "ids": sorted(ids)},
            sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"{_b64(payload)}.{self._mac(payload)}", expires

    def check(self, token: str, close: Closure, *, now: datetime) -> list[str]:
        """The sorted ids `token` binds, or `ConfirmationRefused` saying why not."""
        body, _, mac = token.rpartition(".")
        try:
            payload = _unb64(body)
        except (binascii.Error, ValueError):
            payload = b""
        # Compared before anything in the payload is read, so nothing an attacker wrote
        # is parsed until the key has vouched for it.
        if not body or not hmac.compare_digest(mac, self._mac(payload)):
            raise ConfirmationRefused(
                "this confirmation token was not issued by this memory server, or it was "
                "altered. Nothing was changed. Run the call again without confirm to get "
                "a new preview and token.")
        fields = json.loads(payload)
        if fields["close"] != close:
            verb = "end" if fields["close"] == "ended" else "retire"
            raise ConfirmationRefused(
                f"this token was issued to {verb} memories, and this call asks for "
                f"close={close!r}. Nothing was changed. A preview confirms one closure "
                "only; preview again with the closure you mean.")
        expires = datetime.fromtimestamp(fields["expires"], tz=timezone.utc)
        if as_utc(now) >= expires:
            raise ConfirmationRefused(
                f"this confirmation token expired at {expires.isoformat()}. Nothing was "
                "changed. Run the call again without confirm to see what matches now, "
                "and confirm that preview instead.")
        return list(fields["ids"])
