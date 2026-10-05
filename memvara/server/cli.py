"""`python -m memvara.server` — process startup, and the one place that touches stdio.

The server takes no options besides `--help` and `--version`. Everything is environment
configuration, because that is what an MCP client can actually set: the settings file
gives a command, an argument list and an env block, and the env block is the only part a
user edits per machine. `--help`
exists for the moment someone runs the command by hand to find out why the client says
it failed, and prints the variables rather than a flag list.

`init` is the one subcommand, and it does not weaken that. It is not a way to configure
this process — it writes the settings file the *client* will launch this process from,
which is a different program's configuration and the only place flags could have helped
anyone. See `init.py`.
"""

from __future__ import annotations

import codecs
import io
import os
import sys
from typing import Mapping, Sequence, TextIO, cast

from .. import __version__
from ..core import EmbedderMismatchError
from .config import (
    DEFAULT_SERVER_URL,
    EXAMPLE_CONFIG,
    FEATURES_OFF_BY_DEFAULT,
    ConfigError,
    ServerConfig,
    build_memvara,
)
from .init import init
from .mcp import MemvaraMCPServer

__all__ = ["main"]

#: The decoding error handler for standard input. It puts one NUL in place of each run of
#: bytes that is not UTF-8. JSON allows a NUL nowhere, not even inside a string, so the
#: line then fails to parse and gets the parse error (-32700) any other malformed line
#: gets, at the column of the first NUL, and the server carries on. Replacing the bytes
#: with U+FFFD instead would parse, and store text the client never sent.
_NOT_UTF8 = "memvara-not-utf8"


def _nul_for_each_bad_byte(exc: UnicodeError) -> tuple[str, int]:
    # Registered for decoding only, so `exc` is always a UnicodeDecodeError.
    return "\x00", cast(UnicodeDecodeError, exc).end


codecs.register_error(_NOT_UTF8, _nul_for_each_bad_byte)


def _utf8_stdin() -> tuple[TextIO, io.TextIOWrapper | None]:
    """Standard input, read as UTF-8 whatever the locale says.

    The MCP stdio transport is UTF-8. Python opens standard input in the locale's
    encoding, so under a strict UTF-8 locale one byte that is not UTF-8 raised
    UnicodeDecodeError and ended the server, and under cp1252, a Windows pipe's default,
    "Zürich" sent as UTF-8 was stored as "ZÃ¼rich" (#311). Returns the stream to read,
    and the wrapper this made, if any, which the caller detaches when it is done so that
    closing the wrapper does not close the process's standard input.
    """
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:
        # A text stream with no bytes under it, such as a test's StringIO. There is no
        # encoding to get wrong.
        return sys.stdin, None
    wrapper = io.TextIOWrapper(buffer, encoding="utf-8", errors=_NOT_UTF8)
    return wrapper, wrapper


def _utf8_output(stream: TextIO) -> None:
    """Write one of the process's output streams as UTF-8, as standard input is read.

    The server's startup refusals go to standard error, and a client reads them as UTF-8.
    On Windows, an em dash in a refusal arrived as a cp1252 byte that such a client cannot
    decode (#311). Replies to a client are pure ASCII, but `--help` prints the usage to
    standard output, and it holds em dashes, which an ASCII standard output cannot encode.
    """
    if (isinstance(stream, io.TextIOWrapper)
            and codecs.lookup(stream.encoding).name != "utf-8"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def _feature_defaults() -> str:
    """The `--help` sentence saying which features are on by default, from the table."""
    off = sorted(name.upper() for name in FEATURES_OFF_BY_DEFAULT)
    if not off:
        return "Every feature is on by default."
    listed = off[0] if len(off) == 1 else f"{', '.join(off[:-1])} and {off[-1]}"
    return f"Every feature is on by default except {listed}."


USAGE = f"""\
memvara-mcp {__version__} — Memvara memory as an MCP server over stdio.

This program speaks JSON-RPC on stdin/stdout and is meant to be launched by an MCP
client, not run interactively. Configured entirely by environment:

  MEMVARA_MODE        'local' (default) or 'cloud'. Local opens MEMVARA_DB on this
                     machine. Cloud opens no file at all: it serves the same tools
                     from a hosted deployment over its /v1 API, using MEMVARA_API_KEY
                     or the credential "memvara-mcp login" wrote. Cloud needs
                     pip install "memvara[cloud]", ignores MEMVARA_DB, and refuses
                     MEMVARA_LLM, MEMVARA_EMBEDDER, MEMVARA_READ_W_GRAPH and every
                     model setting below — extraction and embedding run inside the
                     deployment, so a value set here would never be used.
  MEMVARA_SERVER_URL  where cloud mode sends its requests. Unset, it is the address
                     "memvara-mcp login" stored with the key, or
                     {DEFAULT_SERVER_URL}.
  MEMVARA_DB          required in local mode. Path to the SQLite file; created on
                     first use, encrypted unless MEMVARA_FEATURE_ENCRYPTION=0.
                     ':memory:' for a throwaway store that dies with the process.
  MEMVARA_DB_KEY      the key of an encrypted store, as 64 hexadecimal characters.
                     Read when the OS keychain has none (service memvara, account
                     db-key), and before ~/.memvara/db.key, where a key is generated
                     if none exists. A store whose key is lost cannot be read; back
                     it up with `memvara encrypt --export-key`.
  MEMVARA_USER        who this server remembers for. Unset means the whole tenant.
  MEMVARA_TENANT      isolation boundary above the user. Default 'default'.
  MEMVARA_AGENT       narrows further; unset is usually right.
  MEMVARA_SESSION     narrows further still. Memory written here is not visible to
                     other sessions, so leave it unset for durable facts.
  MEMVARA_PROJECT     the repository this memory belongs to, as host/owner/repo (for
                     example github.com/acme/app) or path:<16 hex>. Unset means it is
                     derived from the git remote of the directory the server starts
                     in, so every clone and worktree of one repository shares one
                     project. Outside a git repository there is no project. A fact
                     whose predicate is declared global, such as a preference, is
                     still seen from every project.
  MEMVARA_FEATURE_<NAME>
                     '0' switches one feature off and '1' switches it on.
                     {_feature_defaults()}
                     Tools and arguments: PROFILE=0 hides memory_profile,
                     FORGET_MATCHING=0 hides memory_end_matching and
                     memory_forget_matching, LINKS=0 hides memory_link, and
                     DOCUMENTS=0 hides memory_add_document, memory_get_document,
                     memory_list_documents and memory_delete_document.
                     END_REASON=0 removes the reason and until_reason arguments.
                     QUERY_REWRITE=0 removes the query_rewrite argument and
                     SYNTHESIS=0 removes the synthesize argument, and each also
                     switches that model stage of reading off.
                     METADATA_FILTERS=0 makes memory_search and memory_recall
                     refuse the filters and filepath_prefix arguments.
                     The store: PROJECT_SCOPE=0 stops the project being derived
                     (an explicit MEMVARA_PROJECT still applies). ENCRYPTION=0
                     creates a new store unencrypted; with it on, a new store
                     needs pip install "memvara[encrypt]", and an existing store
                     opens as whatever it already is. RETRIEVAL_CHUNKS=0 keeps
                     each document in one piece instead of splitting it into
                     chunks of about 1,000 characters. INGEST_URLS=0 refuses a
                     document given as a URL, and INGEST_MEDIA=0 refuses images,
                     audio and video. EXPIRY_ERASURE=0 stops erasing the facts
                     whose expires_at has passed, which otherwise happens when the
                     store opens and every hour after.
                     Extraction: EXTRACTION_CHUNKS decides whether a turn over
                     6,000 characters is extracted in pieces, one model call per
                     piece. AGENTIC_EXTRACTION=1 lets the extraction model search
                     the store and propose memories, ends, replacements and links
                     through tools, which the reconciler then applies or refuses.
                     EXTRACTION_GUIDANCE=0 leaves the MEMVARA_EXTRACT_GUIDANCE
                     file out of the prompt, though it is still checked at startup.
                     INDEX_COMMAND, RESEARCH_AGENT, STATUS_LINE, RECALL_MARK and
                     AGENTIC_CAPTURE belong to the plugin and change nothing in
                     this server. They are accepted here so that a misspelled name
                     is refused, as an unknown name is.
  MEMVARA_LLM         'none' (default, offline, extracts only recognised sentence
                     forms), 'anthropic' (needs ANTHROPIC_API_KEY) or 'openai'
                     (needs OPENAI_API_KEY, even for a self-hosted server that
                     ignores it; set OPENAI_BASE_URL to use one). Local mode only.
  MEMVARA_LLM_MODEL   the model 'openai' asks. Unset uses gpt-4.1. A self-hosted
                     server needs the model id it was started with.
  MEMVARA_LLM_MAX_CLAIMS
                     the most claims one 'openai' extraction may return. Unset
                     means no cap, which suits hosted models. Set it for a
                     self-hosted server that constrains decoding to the schema,
                     where a model that repeats itself runs to its token limit.
  MEMVARA_LLM_MAX_TOKENS
                     the most tokens one 'openai' response may use. Unset keeps
                     the backend's own 8,192. A response cut off by this limit is
                     not stored, so set it from a measured response length.
  MEMVARA_LLM_TIMEOUT the seconds one 'openai' extraction may take. Unset keeps the
                     SDK's own 600. A slow self-hosted model may need longer.
  MEMVARA_LLM_EXTRA_BODY
                     a JSON object added to every 'openai' request, such as
                     {{"chat_template_kwargs": {{"enable_thinking": false}}}} for a
                     Qwen3 server.
  MEMVARA_LLM_EXTRACT_SYSTEM
                     the path to a file of extraction instructions for 'openai'
                     that replace the shipped ones.
  MEMVARA_LLM_TERSE_CLAIMS
                     '1' asks 'openai' for a shorter claim, for a slow self-hosted
                     model. Hosted OpenAI refuses the request it makes.
  MEMVARA_EXTRACT_GUIDANCE
                     the path to a TOML file of project guidance (context, include
                     and exclude) added to the extraction prompt of either model.
                     Needs MEMVARA_LLM set, and Python 3.11+.
  MEMVARA_ADVISE_REPLACEMENTS
                     '1' asks the model, after a memory_remember that ended
                     nothing, whether the new fact replaces one of its nearest
                     neighbours, and says so on the receipt. Up to three model
                     calls per write. Needs MEMVARA_LLM set.
  MEMVARA_CLOSED_VOCABULARY
                     '1' refuses a claim the model proposes with a predicate the
                     vocabulary does not know, instead of learning the predicate.
  MEMVARA_EMBEDDER    'hashing' (default, offline, 512-dimensional), 'hashing:<dim>',
                     'local' or 'local:<model>' (needs memvara[local-embed]), or
                     'auto' for whichever of those happens to be installed. A store
                     can only be opened by the embedder that wrote it; if this server
                     refuses to start with a dimension mismatch, this is the variable
                     that fixes it, and the message names the width to give it.
  MEMVARA_PREDICATES  declared predicate vocabularies: shipped pack names, paths to
                     TOML files, or a comma-separated mix, later entries winning.
                     Unset means the built-in vocabulary alone, and a predicate
                     outside it accumulates values instead of superseding them and
                     decays at the slow default. 'engineering' and 'decisions'
                     ship with the package; see docs/INTERNALS.md for the file
                     format. Needs Python 3.11+, where tomllib arrives.
  MEMVARA_READ_ONLY   '1' to hide every tool that writes.
  MEMVARA_READ_W_GRAPH  weight on the graph leg of retrieval, which walks out of
                     the entities the other legs just named. Unset means 0.0, the
                     leg off, which is what every deployment has run. 1.0 gives it
                     the same weight as the vector and lexical legs. A store with
                     no relations in it pays nothing for switching it on.
  MEMVARA_SELECTOR    'local' to answer memory_recall's ranked argument with a small
                     model in this process: no key, no call, nothing sent anywhere.
                     Needs pip install "memvara[rerank]" and downloads about 180 MB
                     of models on first start. Unset, or 'none', serves every ranked
                     read unranked. Refused in cloud mode.
  MEMVARA_SELECTOR_MODEL  a selector model directory written by
                     bench/selector_train.py, in place of the published one. Needs
                     MEMVARA_SELECTOR=local. Refused in cloud mode.
  MEMVARA_ANCHORED'1' to answer only from memories the question is demonstrably
                     about, by default, on all three read tools. A question about
                     an entity this store has never heard of then returns nothing
                     rather than the nearest memory about somebody else. Each call
                     can still pass anchored itself. See docs/DEPLOY.md for what it
                     costs on a question that names no entity.
  MEMVARA_CONFIRM_SECRET  the key that signs the confirmation token of
                     memory_end_matching and memory_forget_matching. Set the same
                     value on every process serving one store. Unset, each process
                     generates its own, which suits a single server.
  MEMVARA_NAT64_PREFIXES
                     your network's own NAT64 prefixes, comma-separated. A
                     document URL whose host resolves into one is checked as the
                     IPv4 address inside it, so a private host behind the gateway
                     is refused. The well-known prefixes are always checked.

The scope above is bound at startup and cannot be changed by a tool call, which is
what stops a model reaching another user's memory.

Rather than writing that block by hand:

  memvara-mcp init --agent claude
                     Writes the client's server block, the memvara skill tree
                     and a project note, with MEMVARA_DB already absolute.
                     `--agent cursor` and `--agent grok` write the same skill
                     into those clients' skill directories. `--skill-only`
                     skips .mcp.json. `memvara-mcp init --help` for its options.

  memvara-mcp login
                     Signs in to a memvara-cloud deployment over the device-code
                     flow and writes an API key to ~/.memvara/credentials.json,
                     for MEMVARA_MODE=cloud. You choose the project in the
                     browser. Needs the `cloud` extra. The same flow as
                     `memvara login`; `memvara-mcp login --help` for its options.

  memvara-mcp --version
                     Prints the version and exits.

Client configuration:

{EXAMPLE_CONFIG}
"""


def main(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None,
         stdin: TextIO | None = None, stdout: TextIO | None = None,
         stderr: TextIO | None = None) -> int:
    """Serve until stdin closes. Returns a process exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if env is None else env
    if stdout is None:
        _utf8_output(sys.stdout)
    if stderr is None:
        _utf8_output(sys.stderr)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    if args:
        # Dispatched before the flags, and by first word rather than by whole argv,
        # because `init` has a command line of its own and this one has none.
        if args[0] == "init":
            return init(args[1:], env=env, stdout=out, stderr=err)
        if args[0] == "login":
            # Imported here, not at module top: `login.py` belongs to the `cloud` extra
            # and requires `httpx`, and this file has to stay importable with no extras
            # installed (CONTRIBUTING.md's "no extras" CI job) for everyone who never
            # calls `login`.
            from .login import login

            return login(args[1:], env=env, stdout=out, stderr=err)
        if args == ["--version"]:
            print(__version__, file=out)
            return 0
        if args in (["--help"], ["-h"]):
            print(USAGE, file=out)
            return 0
        print(f"memvara-mcp: unexpected argument {args[0]!r}\n\n{USAGE}", file=err)
        return 2

    # Bound before the try so the ImportError branch below can ask which mode raised.
    # `from_env` imports nothing lazily today, so this stays None only if it starts to.
    config: ServerConfig | None = None
    try:
        config = ServerConfig.from_env(env)
        memory = build_memvara(config)
    except ConfigError as exc:
        # The client shows this to the user as the reason the server would not start,
        # which is the only moment they are looking. Exit 2, as for a usage error: the
        # invocation was wrong, not the program.
        print(f"memvara-mcp: {exc}", file=err)
        return 2
    except ImportError as exc:
        # `MEMVARA_MODE=cloud` on a machine without `httpx`. The library raised it about a
        # missing package rather than about the configuration, but from here it is a
        # configuration error like any other: this environment names a mode this
        # interpreter cannot run, and the remedy is a pip install. Reported the same way
        # for the same reason as the two below — under stdio the alternative is a
        # traceback in a log nobody reads. `memvara-mcp init --mode cloud` refuses on the
        # same question, so the two commands cannot disagree about whether cloud works
        # here, which is the failure `init` writing a config it never launches invites.
        #
        # **Only for cloud mode.** The local branch imports two optional packages of its
        # own — `sentence-transformers` for MEMVARA_EMBEDDER=local and the anthropic SDK
        # for MEMVARA_LLM=anthropic — and each already catches its own ImportError and
        # raises a ConfigError naming the right extra. One escaping those is a bug, and
        # labelling it "MEMVARA_MODE=cloud cannot start" would send whoever hits it to the
        # wrong variable entirely. Re-raised, so it arrives as what it is.
        if config is None or config.mode != "cloud":
            raise
        print(f"memvara-mcp: MEMVARA_MODE=cloud cannot start a server here. {exc}",
              file=err)
        return 2
    except RuntimeError as exc:
        # The store refuses a file written by a newer Memvara, so that this build cannot
        # write rows the newer one cannot read, and says so clearly. It raises a plain
        # RuntimeError, so this is told apart from a bug by its message, and anything else
        # is re-raised: a traceback is what a bug should produce. Reported like the
        # refusals above, rather than as a traceback in a log the client may not show
        # (#299).
        if "was written by a newer Memvara" not in str(exc):
            raise
        print(f"memvara-mcp: {exc}", file=err)
        return 2
    except EmbedderMismatchError as exc:
        # Not a ConfigError — the library raised it about the store — but from here it is
        # one: this environment names an embedder that cannot read this store, and the
        # remedy is a variable, not a code change. Reported the same way for the same
        # reason, because the alternative under stdio is a traceback in a log the user
        # has to go looking for. The message already carries the store's width and the
        # name of whatever wrote it; all this adds is where to type them.
        print(f"memvara-mcp: {exc}\nFrom this server, set MEMVARA_EMBEDDER to match the "
              "store — 'hashing:<dim>' or 'local:<model>', spelled as above — rather "
              "than editing code. See docs/DEPLOY.md.", file=err)
        return 2

    server = MemvaraMCPServer(memory, read_only=config.read_only,
                              anchored=config.anchored, features_off=config.features_off,
                              **config.scope_kwargs)
    source, made = (stdin, None) if stdin is not None else _utf8_stdin()
    try:
        server.serve(source, out)
    finally:
        # Closing the store matters even on the way out: the vector index is a file this
        # process may have been extending, and other processes share it.
        server.close()
        if made is not None:
            made.detach()
    return 0
