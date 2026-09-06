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

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..core import Memvara
from ..embed import CachedEmbedder, HashingEmbedder
from ..llm import NullLLM
from ..schema import (BUILTIN_PREDICATES, PredicatePackError,
                      PredicateRegistry, load_all_specs)

if TYPE_CHECKING:
    # Imported for the annotation alone. At runtime `memvara.remote.api` reaches back
    # into this module through `remote/creds.py`, so a module-level import here would
    # be a cycle; `build_memvara` imports it inside the branch that needs it.
    from ..remote.api import RemoteMemvara

__all__ = ["ConfigError", "ServerConfig", "build_memvara"]

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}

#: Modes selectable from the environment. "local" is unchanged, decades-old behaviour:
#: a file on disk named by MEMVARA_DB. "cloud" is the device-code auth contract: no
#: MEMVARA_DB, an API key from either the environment or the credentials file written
#: by `memvara-mcp login`, and a `RemoteMemvara` — a client of the `/v1` facade, not
#: an engine over a remote store. `build_memvara` says why that distinction is the
#: whole design.
_MODES = ("local", "cloud")

#: Where `memvara-mcp login` writes what it obtained, and where `from_env` reads it back
#: from when MEMVARA_API_KEY is not set directly. Kept as a module constant because the
#: login command (built separately) has to write exactly this path for this file to find it.
CREDENTIALS_PATH = Path.home() / ".memvara" / "credentials.json"

#: Where a client goes absent MEMVARA_SERVER_URL. `login.py` declares its own copy as
#: `_DEFAULT_SERVER_URL`; leaving that alone is deliberate, since collapsing them is a
#: change to a module this work has no other reason to touch.
DEFAULT_SERVER_URL = "https://app.memvara.dev"

#: Backends selectable from the environment. Anything needing constructor arguments —
#: a custom model, an injected client — is a reason to import `MemvaraMCPServer` and wire
#: it in Python rather than to grow a configuration language here.
_BACKENDS = ("none", "anthropic", "openai")

#: Embedders selectable from the environment, as `kind` or `kind:argument`.
#:
#: The argument is the one exception to the rule above, and it is not a configuration
#: language creeping in — it is the only fact about an *existing* store that cannot be
#: known when this tuple is written. A store's vectors have a width and a model behind
#: them; open it with anything else and `Memvara()` refuses, correctly, before anything
#: writes. The refusal already prints both (`this store holds 512-dimensional vectors,
#: written by hashing:512:3-5`) and `memory.db.embedder.json` records them, so these
#: values are deliberately spelled the way those two already are: `hashing:512` and
#: `local:sentence-transformers/all-MiniLM-L6-v2` are copied, not composed.
_EMBEDDERS = ("hashing", "hashing:<dim>", "local", "local:<model>", "auto")
_EMBEDDER_KINDS = {spec.partition(":")[0] for spec in _EMBEDDERS}
_TAKES_ARGUMENT = {spec.partition(":")[0] for spec in _EMBEDDERS if ":" in spec}

#: What `HashingEmbedder()` and `default_embedder()` both use, restated because
#: `MEMVARA_EMBEDDER=hashing` has to keep meaning the same width forever: it is the width
#: every store written by a deployment with no extras installed already has.
_DEFAULT_DIM = 512

#: Named once, because the field default and the unset-variable default have to be the
#: same string and nothing else can enforce that. A `ServerConfig` built in Python and one
#: read from an environment that sets nothing must open the same store.
_DEFAULT_EMBEDDER = "hashing"

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

    path: str = ""
    tenant: str = "default"
    user: str | None = None
    agent: str | None = None
    session: str | None = None
    read_only: bool = False
    llm: str = "none"
    #: The model name for the "openai" backend. `OpenAILLM(model=...)` defaults to
    #: "gpt-4.1", which names nothing on a self-hosted OpenAI-compatible server (vLLM,
    #: llama.cpp, Ollama's shim) — those need the id the server was started with. The
    #: endpoint itself is not a memvara setting: `OpenAILLM`'s default client construction
    #: reads `OPENAI_BASE_URL` from the SDK's own environment handling.
    llm_model: str | None = None
    #: Cap on the claims array for the "openai" backend, and it only matters for the
    #: self-hosted case `llm_model` describes. A server that constrains decoding to the
    #: schema — llama.cpp and vLLM compile it to a grammar — cannot end a response the
    #: schema still permits to continue, so a model that begins restating itself runs to
    #: its token limit and the reply arrives as truncated JSON that parses as nothing.
    #: Unset leaves the array uncapped, which is right for hosted OpenAI: it closes the
    #: array itself, and its strict mode rejects `maxItems` outright.
    llm_max_claims: int | None = None
    #: Path to a file holding replacement extraction instructions for the "openai"
    #: backend — again the self-hosted case `llm_model` describes. The shipped
    #: `EXTRACT_SYSTEM` ends by saying an empty list is a correct answer and the common
    #: case; a small model reads that as permission and returns nothing on a long turn.
    #: `OpenAILLM.__init__` carries the measurement. A path rather than the text itself,
    #: because a multi-paragraph prompt in an environment variable is unreadable in
    #: `docker inspect` and unmaintainable in a compose file.
    llm_extract_system: str | None = None
    #: Ask the "openai" backend for the shorter claim shape, and again this is only for
    #: the self-hosted case `llm_model` describes. The shipped schema requires every field
    #: on every claim, so a model writes `"when":null,"amount":null,"unit":null` and a
    #: confidence number for each one. On a CPU-hosted model those tokens are most of the
    #: wall time: eight claims are 413 tokens under the shipped schema and 229 under this
    #: one. Two things to know before turning it on. It is a 400 against hosted OpenAI,
    #: whose strict mode requires every declared property in `required`. And it stops
    #: `confidence` being a number the model chose, so every claim lands at 0.5 and a store
    #: written this way ranks differently from one written without it —
    #: `llm.base.self_hosted_claim_schema` carries the detail.
    llm_terse_claims: bool = False
    #: "local" (default) opens MEMVARA_DB on disk, exactly as before this field existed.
    #: "cloud" opens no local file at all; it resolves an API key (MEMVARA_API_KEY, or
    #: the credentials file `memvara-mcp login` writes) and talks to `server_url` instead.
    mode: str = "local"
    server_url: str = DEFAULT_SERVER_URL
    api_key: str | None = None
    #: Which vector space this server's store is opened in. Named rather than discovered,
    #: for the same reason `llm` is: a store outlives the environment it was created in,
    #: and a setting that reads "whatever happens to be installed" makes the store's
    #: identity a property of the machine's package set. `pip install memvara[rerank]`
    #: is the case that proves it — see `build_memvara`.
    embedder: str = _DEFAULT_EMBEDDER
    #: Declared predicate vocabularies: shipped pack names, paths to TOML files, or a
    #: comma-separated mix of both, later entries winning. Empty means the 23 builtins
    #: alone — which is what every server-backed store had before this field existed,
    #: and why a predicate outside them accumulated instead of superseding.
    predicates: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ServerConfig":
        env = os.environ if env is None else env

        mode = (env.get("MEMVARA_MODE") or "local").strip().lower()
        if mode not in _MODES:
            raise ConfigError(
                f"MEMVARA_MODE={mode!r} is not a mode. Use {_one_of(_MODES)}. 'local' is "
                "the default: a MEMVARA_DB file on disk. 'cloud' talks to a memvara-cloud "
                "deployment instead and needs no MEMVARA_DB.")

        path = (env.get("MEMVARA_DB") or "").strip()
        api_key: str | None = None
        server_url = (env.get("MEMVARA_SERVER_URL") or DEFAULT_SERVER_URL).strip() \
            or DEFAULT_SERVER_URL

        if mode == "local":
            if not path:
                raise ConfigError(
                    "MEMVARA_DB is not set, so there is nowhere to remember anything. "
                    "Point it at a file the client can write; it is created on first "
                    "use:\n\n"
                    f"{EXAMPLE_CONFIG}\n\n"
                    "Use MEMVARA_DB=:memory: only for a smoke test — that store is "
                    "discarded when this process exits.")
        else:
            api_key = _optional(env.get("MEMVARA_API_KEY"))
            if api_key is None:
                api_key, credentials_url = _read_credentials()
                server_url = (env.get("MEMVARA_SERVER_URL") or "").strip() \
                    or credentials_url or server_url
            if api_key is None:
                raise ConfigError(
                    "MEMVARA_MODE=cloud needs an API key, and none was found. Set "
                    "MEMVARA_API_KEY, or run \"memvara-mcp login\" to write one to "
                    f"{CREDENTIALS_PATH}.")

        backend = (env.get("MEMVARA_LLM") or "none").strip().lower()
        if backend not in _BACKENDS:
            raise ConfigError(
                f"MEMVARA_LLM={backend!r} is not a backend. Use "
                f"{_one_of(_BACKENDS)}. 'none' is the default and "
                "works offline, but stores only the sentence forms the deterministic "
                "extractor recognises.")

        predicates = (env.get("MEMVARA_PREDICATES") or "").strip()
        if predicates:
            # Read now and discard the result: a typo in a pack name or an unreadable file
            # must be a startup error beside the rest of the configuration, not a
            # surprise at the first write — by which point the process has already
            # accepted facts into the very slots the pack was meant to shape.
            try:
                load_all_specs(predicates)
            except PredicatePackError as exc:
                raise ConfigError(f"MEMVARA_PREDICATES: {exc}") from None

        return cls(
            # `~` is what a human types in a JSON settings file, and nothing else in the
            # launch path will expand it.
            path=(path if path == ":memory:" else os.path.expanduser(path))
                if path else "",
            tenant=(env.get("MEMVARA_TENANT") or "default").strip() or "default",
            user=_optional(env.get("MEMVARA_USER")),
            agent=_optional(env.get("MEMVARA_AGENT")),
            session=_optional(env.get("MEMVARA_SESSION")),
            read_only=_flag(env.get("MEMVARA_READ_ONLY"), "MEMVARA_READ_ONLY"),
            llm=backend,
            llm_model=_optional(env.get("MEMVARA_LLM_MODEL")),
            llm_max_claims=_max_claims(env.get("MEMVARA_LLM_MAX_CLAIMS")),
            llm_extract_system=_optional(env.get("MEMVARA_LLM_EXTRACT_SYSTEM")),
            llm_terse_claims=_flag(
                env.get("MEMVARA_LLM_TERSE_CLAIMS"), "MEMVARA_LLM_TERSE_CLAIMS"),
            embedder=_embedder_spec(env.get("MEMVARA_EMBEDDER")),
            mode=mode,
            server_url=server_url,
            api_key=api_key,
            predicates=predicates,
        )

    @property
    def scope_kwargs(self) -> dict[str, Any]:
        return {"tenant": self.tenant, "user": self.user, "agent": self.agent,
                "session": self.session}


def _max_claims(raw: str | None) -> int | None:
    """A positive integer, or `None` for uncapped.

    Refused at startup rather than clamped, because every wrong value here means
    something an operator would want to know: `0` would forbid every claim and turn
    extraction into a silent no-op, and a typo that fell back to uncapped would leave a
    grammar backend with the failure the cap was set to prevent.
    """
    value = (raw or "").strip()
    if not value:
        return None
    if not (value.isdigit() and int(value) > 0):
        raise ConfigError(
            f"MEMVARA_LLM_MAX_CLAIMS={raw!r} is not a positive integer. Leave it unset "
            "for no cap, which is what hosted models want. Set it only for a self-hosted "
            "server that constrains decoding, where an uncapped array has no legal way to "
            "end a response.")
    return int(value)


#: Most a prompt file may hold. `EXTRACT_SYSTEM` is about 2 KB, so this is generous by a
#: factor of thirty and still small enough that the wrong path is caught rather than
#: loaded. Without a ceiling, `read()` on a mistyped path — a log, a dataset, a model file
#: sitting in the next directory along — pulls the whole thing into memory and then ships
#: it as the system message on every extraction.
_MAX_EXTRACT_SYSTEM_BYTES = 64 * 1024


def _read_extract_system(path: str | None) -> str | None:
    """The replacement extraction instructions named by `MEMVARA_LLM_EXTRACT_SYSTEM`.

    Every way this can go wrong is refused at startup rather than falling back to the
    shipped prompt. A deployment that named this file meant to change what the model is
    told, and quietly not changing it is the failure that looks like success: the server
    starts, extraction runs, and the only symptom is claims that do not arrive.

    "Every way" is wider than it first looks, and the reason is the deployment this
    variable exists for. A self-hosted box keeps prompt files and multi-gigabyte model
    weights within a path typo of each other, so the wrong file here is not hypothetical:
    it is binary (`UnicodeDecodeError`, which is **not** an `OSError` and would otherwise
    escape as a traceback naming no variable), or it is enormous, or it is a directory.
    Each is caught and named.

    Read as `utf-8-sig` so a file written by an editor that emits a byte-order mark does
    not start the system prompt with an invisible character. `str.strip()` would not
    remove it — `\ufeff` is not whitespace — and nothing downstream would show it.

    Only the "openai" backend consults this, so a bad path under `MEMVARA_LLM=anthropic`
    is never reached and never refused. `docs/DEPLOY.md` says so; the alternative is
    validating a file this process has no intention of using.
    """
    if path is None:
        return None
    try:
        size = os.path.getsize(path)
        if size > _MAX_EXTRACT_SYSTEM_BYTES:
            raise ConfigError(
                f"MEMVARA_LLM_EXTRACT_SYSTEM={path!r} is {size} bytes, over the "
                f"{_MAX_EXTRACT_SYSTEM_BYTES} this accepts. A system prompt is kilobytes, "
                "so a file this size is a path pointing at something else.")
        with open(path, encoding="utf-8-sig") as handle:
            text = handle.read().strip()
    except OSError as exc:
        raise ConfigError(
            f"MEMVARA_LLM_EXTRACT_SYSTEM={path!r} cannot be read: {exc}. Unset it to use "
            "the extraction instructions memvara ships.") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"MEMVARA_LLM_EXTRACT_SYSTEM={path!r} is not UTF-8 text: {exc}. This wants the "
            "prompt itself, not a model file.") from exc
    if not text:
        raise ConfigError(
            f"MEMVARA_LLM_EXTRACT_SYSTEM={path!r} is empty. A model told nothing extracts "
            "nothing; unset it to use the extraction instructions memvara ships.")
    return text


def _optional(raw: str | None) -> str | None:
    """An unset variable and one set to the empty string mean the same thing here.

    They do not in general, but a client that writes `"MEMVARA_USER": ""` into its
    settings meant "no user", and binding a scope to the empty-string user instead would
    silently create a second, invisible partition of the store.
    """
    value = (raw or "").strip()
    return value or None


def _one_of(values: Sequence[str]) -> str:
    """`'a', 'b' or 'c'` — so both levers spell their vocabulary the same way."""
    quoted = [repr(v) for v in values]
    return f"{', '.join(quoted[:-1])} or {quoted[-1]}"


def _embedder_spec(raw: str | None) -> str:
    """Validate `MEMVARA_EMBEDDER` at startup, and normalise the case of its kind.

    Only the kind is lowercased. The argument is a width or a model id, and
    `local:sentence-transformers/all-MiniLM-L6-v2` is case-sensitive — it is copied out
    of the store's own fingerprint, so mangling it would break the one value a
    recovering operator has in front of them.

    Checked here rather than in `build_memvara` for the same reason `MEMVARA_LLM` is: a
    typo should be a startup error beside the rest of the configuration, not a stack
    trace from somewhere inside the constructor.
    """
    value = (raw or _DEFAULT_EMBEDDER).strip()
    kind, sep, argument = value.partition(":")
    kind = kind.lower()
    if (kind not in _EMBEDDER_KINDS or (sep and kind not in _TAKES_ARGUMENT)
            or (sep and not argument)):
        raise ConfigError(
            f"MEMVARA_EMBEDDER={value!r} is not an embedder. Use "
            f"{_one_of(_EMBEDDERS)}. 'hashing' is the default: it needs nothing, runs "
            "offline, and is what a store written without any extra installed already "
            "holds. 'auto' is the old behaviour — whichever embedder happens to be "
            "installed — and is a choice about your store, not about your packages.")
    if kind == "hashing" and argument and not (argument.isdigit() and int(argument) > 0):
        raise ConfigError(
            f"MEMVARA_EMBEDDER={value!r} does not name a width: {argument!r} is not a "
            "positive integer. The number is the one the store already reports — "
            "'this store holds N-dimensional vectors' in the error that sent you here, "
            "or \"dim\" in memory.db.embedder.json.")
    return f"{kind}:{argument}" if sep else kind


def _read_credentials() -> tuple[str | None, str | None]:
    """Read the api_key and server_url `memvara-mcp login` wrote, or (None, None).

    Any way this file could fail to be a usable credential — missing, unreadable, not
    JSON, no api_key — is treated the same as "not logged in yet" rather than raised
    directly, so the caller's ConfigError naming "memvara-mcp login" is the one message
    the user sees.
    """
    try:
        data = json.loads(CREDENTIALS_PATH.read_text())
    except (OSError, ValueError):
        return None, None
    api_key = data.get("api_key") if isinstance(data, dict) else None
    server_url = data.get("server_url") if isinstance(data, dict) else None
    if not isinstance(api_key, str) or not api_key:
        return None, None
    return api_key, server_url if isinstance(server_url, str) and server_url else None


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
    except Exception as exc:
        # Not just ImportError: `AnthropicLLM` raises ValueError when no key is set,
        # and the SDK has its own error for a client it cannot construct. All of them
        # mean the same thing to the operator - this backend cannot start - and all of
        # them should arrive as the startup ConfigError beside the rest of the
        # configuration rather than as a raw traceback out of build_memvara.
        raise ConfigError(
            f"MEMVARA_LLM=anthropic needs the anthropic SDK and a key: {exc}") from exc


def _openai(model: str | None, max_claims: int | None = None,
            extract_system: str | None = None, terse: bool = False) -> Any:
    # Imported here so the default offline configuration never touches the optional SDK.
    from ..llm.openai import OpenAILLM

    system = _read_extract_system(extract_system)
    try:
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if max_claims is not None:
            kwargs["max_claims"] = max_claims
        if system is not None:
            kwargs["extract_system"] = system
        # Sent only when asked for, so a server that never set the variable makes the same
        # request it made before this option existed — byte for byte, including the schema
        # object identity `test_the_shared_claim_schema_stays_uncapped_for_the_hosted_path`
        # pins.
        if terse:
            kwargs["terse"] = True
        return OpenAILLM(**kwargs)
    except Exception as exc:
        # Deliberately wider than ImportError. `openai.OpenAI()` refuses to construct
        # without a key, and that refusal is the SDK's own error rather than an import
        # failure - so the self-hosted case this backend exists for (OPENAI_BASE_URL at
        # a local vLLM, which needs no real key) used to crash out of build_memvara with
        # a traceback naming neither memvara variable.
        raise ConfigError(
            f"MEMVARA_LLM=openai needs the openai SDK, and the SDK needs OPENAI_API_KEY "
            f"set even when OPENAI_BASE_URL points at a server that ignores it: {exc}"
        ) from exc


def _local(model: str) -> Any:
    # Imported here so a hashing deployment — the default — never pays the torch import.
    from ..embed.local import LocalEmbedder

    try:
        return CachedEmbedder(LocalEmbedder(model) if model else LocalEmbedder())
    except ImportError as exc:
        raise ConfigError(f"MEMVARA_EMBEDDER={model or 'local'} needs "
                          f"sentence-transformers: {exc}") from exc


def _embedder(spec: str) -> Any:
    """Build the embedder named by `spec`, or `None` to let `Memvara()` pick one.

    `None` is what `auto` means, and it routes through `Memvara.__init__` rather than
    calling `default_embedder()` here on purpose: 'auto' is defined as "the constructor's
    own default", so there is one implementation of that default and not two that can
    drift.
    """
    kind, _, argument = spec.partition(":")
    if kind == "auto":
        return None
    if kind == "local":
        return _local(argument)
    return CachedEmbedder(
        HashingEmbedder(dim=int(argument) if argument else _DEFAULT_DIM))


def _llm(config: ServerConfig) -> Any:
    """Build the backend `config` names, and refuse to guess at one it does not.

    Every branch is explicit and the fallthrough raises, because the alternative
    silently routes an unrecognised backend to whichever one happens to be last:
    a value added to `_BACKENDS` without a branch here would have started a server
    that extracts through a vendor the operator never named, with no error to see.
    `from_env` already rejects anything outside `_BACKENDS`, so this is unreachable
    from the environment and exists to keep the two lists honest with each other.
    """
    if config.llm == "none":
        return NullLLM()
    if config.llm == "anthropic":
        return _anthropic()
    if config.llm == "openai":
        return _openai(config.llm_model, config.llm_max_claims,
                       config.llm_extract_system, config.llm_terse_claims)
    raise ConfigError(
        f"MEMVARA_LLM={config.llm!r} is listed in _BACKENDS but _llm() has no branch "
        "for it, so this server cannot say which model it would extract with. This is "
        "a wiring bug in memvara, not a mistake in your configuration.")


def _registry(config: ServerConfig) -> PredicateRegistry | None:
    """The vocabulary this server declares, or None to take the builtins alone.

    `None` rather than an empty registry on purpose: `Memvara()` builds its own default
    when given nothing, and handing it one built here would duplicate that decision in a
    second place.
    """
    if not config.predicates:
        return None
    return PredicateRegistry(
        specs=BUILTIN_PREDICATES + load_all_specs(config.predicates))


#: Set under cloud mode, these name a subsystem that does not run in this process.
#: Extraction and embedding happen inside the deployment, so the value would be read,
#: accepted and ignored — and an operator who set `MEMVARA_LLM=anthropic`, saw a server
#: start and believed their writes were being extracted by a model has been told something
#: false by a program that stayed silent. Refused, with the variable named.
#: `(field, default, variable, noun)`. The noun is carried rather than reused from
#: `field` because `llm_model` does not read as one in a sentence, and the message is
#: the whole point of the refusal.
_SERVER_SIDE_UNDER_CLOUD = (
    ("llm", "none", "MEMVARA_LLM", "llm"),
    ("embedder", _DEFAULT_EMBEDDER, "MEMVARA_EMBEDDER", "embedder"),
    ("llm_model", None, "MEMVARA_LLM_MODEL", "extraction model"),
    ("llm_max_claims", None, "MEMVARA_LLM_MAX_CLAIMS", "claim cap"),
    ("llm_extract_system", None, "MEMVARA_LLM_EXTRACT_SYSTEM", "extraction prompt"),
    ("llm_terse_claims", False, "MEMVARA_LLM_TERSE_CLAIMS", "claim shape"),
)


def build_memvara(config: ServerConfig) -> "Memvara | RemoteMemvara":
    """Open the memory this server speaks for: a local store, or a hosted deployment.

    In "local" mode the scope goes to the constructor as well as to `Memvara.scope()`
    later, because the tenant decides which learned predicate vocabulary is rehydrated at
    open — a server that scoped only its calls would classify every predicate again on
    every launch.

    In "cloud" mode there is no local file and no local engine. This returns a
    `RemoteMemvara`: a client of the `/v1` facade that turns each library call into one
    request and hydrates the reply into the same dataclasses, which is why one MCP tool
    table can serve either. It is **not** a `Memvara` over a `RemoteStore`, and that
    distinction is the whole decision — the engine calls `put_claim`, `lexical_search`
    and `competing_claims` on every turn and the facade has an endpoint for none of them,
    so a server built that way would start, list fourteen tools and fail on the first one
    a model reached for. See `docs/OPEN-CORE.md` for which side of the line each seam is
    on.
    """
    if config.mode == "cloud":
        from ..remote.api import RemoteMemvara

        if config.api_key is None:
            # Reachable only by constructing a ServerConfig directly in Python, bypassing
            # from_env's own check (which never returns mode="cloud" without an api_key) —
            # the equivalent of the "no MEMVARA_DB" path below, given the same error type
            # for the same reason: the environment (or here, the Python caller) named a
            # configuration that cannot open a store.
            raise ConfigError(
                "build_memvara() was given a ServerConfig(mode='cloud') with no "
                "api_key. ServerConfig.from_env() never produces this combination; "
                "a caller constructing ServerConfig directly must set api_key too.")

        for field, default, variable, noun in _SERVER_SIDE_UNDER_CLOUD:
            chosen = getattr(config, field)
            if chosen != default:
                raise ConfigError(
                    f"{variable}={chosen!r} does not apply under MEMVARA_MODE=cloud. "
                    f"The {noun} runs inside the deployment, so this process would "
                    "read the setting and never use it. Unset it, or use "
                    "MEMVARA_MODE=local with MEMVARA_DB if you want this machine to do "
                    f"the work. \"memory_stats\" reports the deployment's own {noun}.")

        return RemoteMemvara(
            api_key=config.api_key,
            base_url=config.server_url,
            **config.scope_kwargs,
        )
    return Memvara(
        config.path,
        # Passed explicitly even when it is the default: `Memvara()` warns about a missing
        # extraction model, and that warning goes to stderr, which under stdio nobody
        # reads. `memory_stats` and the note on every lossy write say it where the model
        # and the user can actually see it.
        llm=_llm(config),
        # Explicit for a sharper reason than `llm`: the embedder decides whether this
        # store can be opened at all. Left unset, `Memvara()` calls `default_embedder()`,
        # which returns a sentence-transformers model as soon as that package is
        # importable — and `memvara[rerank]` installs one, because a cross-encoder is
        # one. So `pip install memvara[rerank]` into a working deployment used to make
        # this server refuse to open its own store on the next launch, with a dimension
        # mismatch nobody connected to reranking. The error was right and its advice —
        # pass the original embedder rather than migrating — had no door to come through
        # here. `MEMVARA_EMBEDDER` is that door, and naming a default rather than
        # discovering one is what keeps the store's vector space out of `pip`'s hands.
        embedder=_embedder(config.embedder),
        # The door this server did not have. `Memvara` has always taken a registry, but
        # an MCP client can only set environment variables, so a server-backed store was
        # pinned to the builtins and everything outside them accumulated silently.
        registry=_registry(config),
        **config.scope_kwargs,
    )
