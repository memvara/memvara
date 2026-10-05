"""Client for a hosted memvara-cloud deployment.

This package is a client of the `/v1` facade. The engine runs on the server, and this
client sends it one request per method call. See `docs/OPEN-CORE.md` for why the engine is
not run over a remote store instead.
"""
from .errors import (
    AuthError, Conflict, InvalidRequest, LegalHold, NotFound, QuotaExhausted,
    RateLimited, ReadOnly, RemoteError, ScopeError, ServerError,
)

__all__ = [
    "RemoteError", "AuthError", "ScopeError", "NotFound", "Conflict",
    "QuotaExhausted", "RateLimited", "LegalHold", "ReadOnly", "InvalidRequest",
    "ServerError",
]
