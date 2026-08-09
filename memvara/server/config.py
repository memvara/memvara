"""Where the server's scope and store come from: the client's environment, and only that.

An MCP stdio server is launched by the client with an environment block the user wrote
in their own settings file. That block is the credential — it says which database and
which user this process speaks for — and reading scope from anywhere else, in particular
from tool arguments, would hand the model the ability to address other people's memory.

Everything here fails loudly. A memory server that starts against the wrong store, or
against a throwaway in-memory one, looks identical to a working server for exactly as
long as it takes the user to notice that nothing is ever remembered; the client shows a
failed launch and the message below immediately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from ..core import Memvara
from ..llm import NullLLM

__all__ = ["ConfigError", "ServerConfig", "build_memvara"]

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}

#: Backends selectable from the environment. Anything needing constructor arguments —
#: a custom model, an injected client — is a reason to import `MemvaraMCPServer` and wire
#: it in Python rather than to grow a configuration language here.
_BACKENDS = ("none", "anthropic")

EXAMPLE_CONFIG = """\
{
  "mcpServers": {
    "memvara": {
      "command": "python3",
      "args": ["-m", "memvara.server"],
      "env": {
        "MEMVARA_DB": "/absolute/path/to/memory.db",
        "MEMVARA_USER": "your-name"
      }
    }
  }
}"""


class ConfigError(Exception):
    """The environment does not describe a server that could work."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Everything the process needs to know, resolved once at startup."""

    path: str
    tenant: str = "default"
    user: str | None = None
    agent: str | None = None
    session: str | None = None
    read_only: bool = False
    llm: str = "none"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ServerConfig":
        env = os.environ if env is None else env
        path = (env.get("MEMVARA_DB") or "").strip()
        if not path:
            raise ConfigError(
                "MEMVARA_DB is not set, so there is nowhere to remember anything. Point "
                "it at a file the client can write; it is created on first use:\n\n"
                f"{EXAMPLE_CONFIG}\n\n"
                "Use MEMVARA_DB=:memory: only for a smoke test — that store is discarded "
                "when this process exits.")

        backend = (env.get("MEMVARA_LLM") or "none").strip().lower()
        if backend not in _BACKENDS:
            raise ConfigError(
                f"MEMVARA_LLM={backend!r} is not a backend. Use "
                f"{' or '.join(repr(b) for b in _BACKENDS)}. 'none' is the default and "
                "works offline, but stores only the sentence forms the deterministic "
                "extractor recognises.")

        return cls(
            # `~` is what a human types in a JSON settings file, and nothing else in the
            # launch path will expand it.
            path=path if path == ":memory:" else os.path.expanduser(path),
            tenant=(env.get("MEMVARA_TENANT") or "default").strip() or "default",
            user=_optional(env.get("MEMVARA_USER")),
            agent=_optional(env.get("MEMVARA_AGENT")),
            session=_optional(env.get("MEMVARA_SESSION")),
            read_only=_flag(env.get("MEMVARA_READ_ONLY"), "MEMVARA_READ_ONLY"),
            llm=backend,
        )

    @property
    def scope_kwargs(self) -> dict[str, Any]:
        return {"tenant": self.tenant, "user": self.user, "agent": self.agent,
                "session": self.session}


def _optional(raw: str | None) -> str | None:
    """An unset variable and one set to the empty string mean the same thing here.

    They do not in general, but a client that writes `"MEMVARA_USER": ""` into its
    settings meant "no user", and binding a scope to the empty-string user instead would
    silently create a second, invisible partition of the store.
    """
    value = (raw or "").strip()
    return value or None


def _flag(raw: str | None, name: str) -> bool:
    value = (raw or "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(
        f"{name}={raw!r} is not a boolean. Use one of "
        f"{', '.join(sorted(_TRUE))} or {', '.join(sorted(_FALSE - {''}))}.")


def _anthropic() -> Any:
    # Imported here so the default offline configuration never touches the optional SDK.
    from ..llm.anthropic import AnthropicLLM

    try:
        return AnthropicLLM()
    except ImportError as exc:
        raise ConfigError(
            f"MEMVARA_LLM=anthropic needs the anthropic SDK and a key: {exc}") from exc


def build_memvara(config: ServerConfig) -> Memvara:
    """Open the store this server speaks for.

    The scope goes to the constructor as well as to `Memvara.scope()` later, because the
    tenant decides which learned predicate vocabulary is rehydrated at open — a server
    that scoped only its calls would classify every predicate again on every launch.
    """
    return Memvara(
        config.path,
        # Passed explicitly even when it is the default: `Memvara()` warns about a missing
        # extraction model, and that warning goes to stderr, which under stdio nobody
        # reads. `memory_stats` and the note on every lossy write say it where the model
        # and the user can actually see it.
        llm=NullLLM() if config.llm == "none" else _anthropic(),
        **config.scope_kwargs,
    )
