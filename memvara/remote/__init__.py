"""Client for a hosted memvara-cloud deployment.

`memvara.store.remote` is a `Store` — the low-level surface the *engine* calls. This
package is the other seam: a client of the `/v1` facade, where the engine runs
server-side. See `docs/OPEN-CORE.md` for why the two do not converge.
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
