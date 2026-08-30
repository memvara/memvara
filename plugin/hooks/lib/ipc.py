"""Addressing and framing for the recall daemon. Shared by both ends.

The socket's *name* does most of the safety work here, so it is worth saying why it is
built the way it is.

It contains a digest of three things: the store the daemon opened, the source code it
opened it with, and the host it was opened for.

* **The store**, because a daemon is a warm handle on one specific database. A second
  project pointing at a different `MEMVARA_DB` must not reach it — it would be answering
  one store's questions out of another store's memory, which is a privacy failure, not a
  cache miss.
* **The code**, because a long-lived process keeps running whatever it was started with.
  Edit a hook and the daemon serves the old logic indefinitely, and nothing looks wrong.
  Folding a digest of the sources into the name means changed code simply addresses a
  different socket: the new client starts a new daemon, and the old one idles out and
  exits on its own. No version negotiation, no restart command to remember, no way to be
  silently served by stale code.
* **The host**, because these bytes are vendored into a plugin repository per coding
  client. Two of them installed side by side share a machine, and can share a store and a
  hook tree byte for byte, so without this the first daemon to bind serves both clients
  -- and since the bound host is what decides which config files `server_env` reads, the
  two need not even have resolved the same store to have collided on one address.

The directory is `0700` and the socket `0600`. A unix socket carrying recall output is a
read interface to everything the user has ever stored, and the default umask would have
left it readable by every account on the machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path
import socket

# `pathlib` is deliberately absent. Importing it costs 10.5ms measured, against a client
# whose entire budget is ~30ms, and every path here is a string join and a stat. `open.py`
# still uses it freely — that module is only reached on the fallback path, where 10ms is
# already lost in the noise of a 148ms in-process query.
_HOME = os.path.expanduser("~")

#: Private by construction: created 0700, and re-chmod'd on every call because an
#: existing directory from an older version may predate that rule.
RUNTIME_DIR = os.path.join(_HOME, ".memvara", ".hooks", "run")

#: Files whose contents decide what a daemon actually does. A change to any of them must
#: strand the old daemon rather than let it keep serving.
CODE_FILES = ("daemon.py", "lib/ipc.py", "lib/open.py", "recall.py",
              "run.py", "core/host.py", "core/envelope.py", "hosts/claude.py")

#: Set in the environment of the `claude -p` child that `capture.py` spawns to mine a turn.
#: A hook that finds it is running underneath an extraction rather than in front of a
#: person, and must stand down.
#:
#: It lives here, in the one module every hook already imports, because the alternative is
#: the literal string in each of them -- and a sentinel that has drifted apart across four
#: files fails by doing nothing, which is the failure this whole guard exists to stop.
#: `lib.extract` imports it from here rather than declaring its own.
CAPTURE_SENTINEL = "MEMVARA_CAPTURE_ACTIVE"


def under_extraction() -> bool:
    """Whether this hook is running inside the extractor's own child process.

    `capture.py` launches that child with `--settings '{"hooks":{}}'`, and the comment
    there long claimed the empty hook set was what kept the child inert. It is not: that
    clears the hooks a *settings file* declares and does not touch the ones a **plugin**
    registers, so the child ran this plugin's hooks like any other session. Measured rather
    than assumed -- a `claude -p` run fires `SessionStart` and `UserPromptSubmit`, both
    confirmed with a marker file, and `recall-sample.log` caught the result in the act:
    41 of 77 sampled prompts were the extractor's own "Extract durable facts from the
    exchange below", each answered with a retrieval query and a standing block.

    Two costs, and the second is the one that matters. A retrieval query is spent per
    extraction against an allowance that is not per-session. And the child gets a session
    id it has never used before, so nothing is deduplicated and the whole standing block is
    injected -- into the one prompt whose entire job is to decide which sentences in front
    of it are facts worth storing. `capture.py` already passes `injected` to `triples()` to
    stop the store's own output being mined back in as new; letting the child recall in the
    first place hands that filter a problem it should never have been given.
    """
    return bool(os.environ.get(CAPTURE_SENTINEL))


def _alert_path() -> str:
    """Where a failed `claude -p` leaves word for the next prompt to relay.

    Built fresh on every call rather than cached at import, the way `store_key` and
    `log_line` already read `_HOME` -- not `RUNTIME_DIR`'s way, which bakes it into a
    constant. The distinction is not style: this repository's own test fixture re-homes
    `ipc._HOME` for the whole suite, and a frozen path would go on reading and writing the
    real `~/.memvara` from every test that touches it, silently, since a test writing to
    the developer's own machine looks exactly like one that passed.

    Beside the logs, not in the plugin: the plugin directory is replaced wholesale on
    update, and a file that disappears on upgrade would clear every outstanding alert
    along with it. A file rather than the log itself, because `capture.log` is an
    append-only record and this has to answer one question -- "is capture still broken,
    and since when did we last say so" -- without re-reading everything ever written to it.
    """
    return os.path.join(_HOME, ".memvara", ".hooks", "capture-alert.json")


def _read_json_file(path: str) -> dict:
    """A dict from a JSON file, or `{}` for anything short of one -- missing, unreadable,
    corrupt, or holding some other JSON shape entirely. Shared by `_read_alert` and
    `_read_notified_alert`: same file shape, same failure handling, same reason for it --
    only the path differs.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_file_atomic(path: str, data: dict, tmp_prefix: str,
                            log_name: str = "") -> None:
    """Write `data` to `path` via a sibling temp file and `os.replace`, so a concurrent
    reader never sees a truncated payload. Shared by `_write_alert` and
    `_write_notified_alert` -- the mechanism is identical between the two files; it is WHO
    writes each one and WHAT race each accepts that differs, which is why that reasoning
    stays on each wrapper's own docstring rather than living here.

    `log_name`, when given, is the one thing this helper does that a plain `open(path, "w")`
    would not have needed: a line to that log on a write failure. Silent is fine for a
    write's FIRST failure -- an occasional missed update is a tradeoff this repository
    already accepts for `capture-alert.json`'s siblings -- but a write that keeps failing
    forever is a different thing, and for at least one caller (`due_alert_for_model`) a
    permanently failing write means the state it exists to update never advances, so the
    same notice repeats on every single prompt for as long as the write keeps losing --
    exactly the per-turn repetition that function exists to prevent, with nothing anywhere
    saying why it stopped working. Omitted (empty string) for `_write_alert`, unchanged from
    before this helper existed, since that failure mode was already reviewed and accepted
    on its own terms.
    """
    import tempfile

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=tmp_prefix)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            if log_name:
                log_line(log_name, f"write failed: {os.path.basename(path)}")
    except OSError:
        if log_name:
            log_line(log_name, f"write failed: {os.path.basename(path)}")


def _read_alert() -> dict:
    return _read_json_file(_alert_path())


def _write_alert(data: dict) -> None:
    """This file's only writer that has a payload to write.

    `clear_capture_alert` is the OTHER thing that touches this file, and does not call
    this function -- it does a plain `os.unlink`, which needs no atomicity dance because
    an unlink has no partial-write state to leave behind. The two are still genuinely
    concurrent with each other: `raise_capture_alert` (the async extraction child) and
    `clear_capture_alert` (the same child, the next time it succeeds) can each run while
    the other is mid-call, with no lock between them -- `capture.py` says extraction takes
    12-14s and hands the turn back immediately, so an extraction still finishing while
    another one starts is the ordinary case, not a rare one.

    What atomicity fixes here is narrower: a reader opening the file mid-write and getting
    a truncated payload. Writing to a sibling temp file and `os.replace`-ing it over the
    target means a concurrent `_read_alert` sees either the whole old file or the whole new
    one, never a half-written mix that `json.load` would raise on and `_read_alert` would
    then read back as "no alert" -- silently hiding an active failure for one prompt.

    What neither this nor `clear_capture_alert`'s unlink fixes: the two can still race with
    each other, so whichever lands last wins outright rather than merging -- a
    `clear_capture_alert` from one extraction can land after a `raise_capture_alert` from
    a second, failing one, dropping a real alert; the reverse can resurrect a resolved one
    for one more prompt. Closing that needs a lock around both writers, which is real
    machinery for what this is: an occasional missed or resurrected banner between two
    concurrent extractions, corrected the moment either one runs again. Losing a fact would
    be the larger bug; losing a few seconds of an FYI banner's accuracy is the trade this
    file already makes for `capture.log` and the session state next to it.
    """
    _write_json_file_atomic(_alert_path(), data, ".capture-alert-")


def raise_capture_alert(reason: str) -> None:
    """Record that `claude -p` just failed, in the words `lib.extract` already logged.

    Called from every failure exit in `_payload` except the recursion guard at the very
    top, which fires inside the extraction child itself and is the guard working, not a
    failure -- alerting on it would mean every successful stand-down looked like an
    outage.

    Overwrites whatever was here before unconditionally. There used to be a clock in this
    file too -- report once, then suppress the same reason for six hours -- and it meant a
    prompt sent minutes after the throttle window opened got the plain banner back with no
    word that anything was wrong, which reads as "fixed" to someone who was never told
    otherwise. `due_capture_alert` no longer owns a schedule to consult, so there is nothing
    here left to preserve between failures.
    """
    _write_alert({"reason": reason})


def clear_capture_alert() -> None:
    """Called the first time `claude -p` succeeds after failing. A no-op if it never failed.

    Also resets what `due_alert_for_model` has already told the model, at the moment
    capture actually recovers rather than lazily the next time something happens to call
    it while nothing is failing. `due_alert_for_model` has its own defensive reset for
    that same case, but relying on it alone would leave a gap: a recovery followed
    immediately by a second, unrelated failure with the same reason text, with no prompt
    landing in between to trigger the lazy path, would read as "already told" -- exactly
    the stale-positive `raise_capture_alert`'s own docstring describes for the
    human-visible banner, just on the model-facing side instead. The `.get` guard keeps
    this a true no-op on the common path where capture has never failed at all -- no
    write, not even one that would land on an already-empty file.
    """
    try:
        os.unlink(_alert_path())
    except OSError:
        pass
    if _read_notified_alert().get("reason"):
        _write_notified_alert({})


def due_capture_alert() -> str:
    """The clause to add to this prompt's banner, or `""` when nothing is due.

    Read unconditionally on every prompt -- an `open()` that raises `FileNotFoundError` on
    an install where capture has never failed, which is the common case and costs one
    failed syscall rather than a stat-then-open pair that would cost two and still race.

    Every prompt, for as long as `capture-alert.json` names a reason -- no reminder clock,
    no suppression window. There used to be one, on the reasoning that repeating the same
    line is noise a person learns to stop reading. Measured against what it actually cost:
    a person watching one banner every few minutes during an outage read the SAME banner
    dozens of times regardless, since recall's own message rides beside it and changes on
    its own schedule -- the alert clause was the only part of that line that ever silently
    disappeared and reappeared, on a clock nothing on screen explained. Silence for six
    hours reads as "this got fixed," not as "already told you," and the six hours where it
    stayed wrong was chosen to match a channel -- `capture.log` -- that nobody was actually
    reading in real time. The channel that matters is exactly the one this function feeds.
    """
    data = _read_alert()
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason:
        return ""
    return f"capture failing: {reason}"


def _notified_alert_path() -> str:
    """Where the reason last handed to the MODEL as context is recorded.

    A sibling of `_alert_path()`, not the same file: `due_capture_alert` is read
    unconditionally on every prompt for the human-visible banner, by design -- see its own
    docstring for why that stopped being throttled. `due_alert_for_model` answers a
    different question -- "has the model already been told about *this* failure" -- and
    needs its own memory to answer it, or every prompt would re-tell the model the same
    reason for as long as it stays broken, moving the exact repetition just removed from
    the terminal banner into the model's own replies instead.
    """
    return os.path.join(_HOME, ".memvara", ".hooks", "capture-alert-notified.json")


def _read_notified_alert() -> dict:
    return _read_json_file(_notified_alert_path())


def _write_notified_alert(data: dict) -> None:
    """Atomic write, same mechanism as `_write_alert` -- but NOT the single-writer file an
    earlier version of this docstring claimed.

    Two things write here, from two different, genuinely concurrent processes: `recall.py`
    calls `due_alert_for_model()` synchronously on every prompt, and `clear_capture_alert()`
    -- called from the async extraction child, the same one `raise_capture_alert` runs from
    -- resets this file the moment capture recovers. Those two can race exactly the way
    `raise_capture_alert`/`clear_capture_alert` already race on `capture-alert.json`: each
    does its own read-then-conditional-write with no lock between them, so a recovery's
    reset can land on a stale read and silently no-op while a fresh notify write is still in
    flight, or two overlapping `recall.py` processes can clobber each other's write outright.
    The accepted trade is identical to that sibling file's: whichever write lands last wins,
    an occasional stale "already told" or an occasional extra retelling, corrected the next
    time either side runs again. Closing it needs the same lock this file's neighbor already
    declines for the same reason -- real machinery for what is, at worst, one repeated or
    one skipped sentence in a reply.
    """
    _write_json_file_atomic(_notified_alert_path(), data, ".capture-alert-notified-",
                            log_name="recall")


def due_alert_for_model() -> str:
    """Context to hand the model when the failing reason is new since it was last told, or `""`.

    This is not the six-hour throttle `due_capture_alert` had removed from it -- that was a
    clock silently hiding a still-active failure from a person watching a banner that should
    have been a live indicator every time it was read. This is a value comparison, not a
    clock: it tells the model once per distinct reason, so a person's actual conversation
    is not interrupted on every single turn by a repeat of the same sentence for however
    many days a failure stays unresolved. The human-visible banner from `due_capture_alert`
    is untouched by this and keeps firing on every prompt exactly as before -- only the copy
    that reaches the model through `hookSpecificOutput.additionalContext` is deduplicated.

    A cleared alert resets what was last told, not just what is currently active: a second,
    unrelated failure that happens to produce the same reason text as an earlier,
    since-fixed one is a new event to tell the model about, not a repeat of the old one.
    """
    reason = _read_alert().get("reason")
    reason = reason if isinstance(reason, str) and reason else ""
    told = _read_notified_alert().get("reason")
    told = told if isinstance(told, str) else ""

    if not reason:
        if told:
            _write_notified_alert({})
        return ""
    if reason == told:
        return ""
    _write_notified_alert({"reason": reason})
    return (
        f"Memvara: memory capture just started failing ({reason}). "
        "Mention this to the user once in your reply."
    )


def with_alert(text: str, alert: str) -> str:
    """A status line, with word of a failing extractor riding along on it.

    `capture.py` runs `async` and cannot print anything of its own -- the client discards
    an async hook's output entirely, which is why its whole account moved to `capture.log`
    in the first place. Nobody reads that on a schedule, so a `claude -p` that has been
    failing for hours says nothing anyone sees until a hook that speaks -- `recall.py` on
    every prompt, `session_start.py` once per session -- relays it.

    Shared rather than defined once per hook: both call sites shadow `emit_json` locally
    right after computing `alert`, so every `systemMessage` that file already prints picks
    this up with no per-call-site wrapping to remember or forget. A first version of this
    threaded `_with_alert(status(...), alert)` through five separate call sites in
    `recall.py` by hand; only one of the five was ever covered by a test, and a sixth site
    added later without the wrap would have printed a perfectly valid banner and failed
    nothing -- the exact "guard nobody can count" shape this repository's CLAUDE.md warns
    against. Shadowing removes the chance to forget rather than relying on remembering.
    """
    return f"{text} · {alert}" if alert else text


#: How long a client waits. Generous next to a 6ms query and mean next to a 148ms cold
#: fallback: past this the daemon is wedged and the in-process path is the faster answer.
CLIENT_TIMEOUT_SEC = 2.0

#: A daemon with no client for this long has outlived its session and exits. This is what
#: stops abandoned processes accumulating after Claude Code quits, since nothing sends a
#: shutdown on exit.
IDLE_TIMEOUT_SEC = 30 * 60


#: The brand mark, as one character.
#:
#: `public/brand/mark-dark.svg` is two arcs on one axis meeting at a node -- valid time
#: closed, transaction time still open, which is the product's whole idea. A
#: `systemMessage` is a string the client renders into its own terminal UI, so there is no
#: image channel to put that in: whatever stands in for it has to be one glyph.
#:
#: BOWTIE is the closest honest one. Two strokes meeting at a node is the mark's own
#: geometry, and it is the relational-algebra join -- which is what a store of facts about
#: one subject does. It is BMP rather than an emoji on purpose: a glyph a terminal font
#: lacks renders as a tofu box, which is worse than no mark at all, and BMP maths symbols
#: are carried nearly everywhere a monospace font is.
MARK = "\u22c8"


def status(text: str) -> str:
    """The one line a person watching the terminal actually sees.

    Composed here rather than at each call site because there were eight of them, every one
    repeating both the mark and the word. A status line that says `Memvara` in seven places
    and something else in the eighth is the kind of drift nobody notices until a screenshot.
    """
    return f"{MARK} Memvara \u00b7 {text}"


def runtime_dir() -> str:
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    try:
        os.chmod(RUNTIME_DIR, 0o700)
    except OSError:
        pass
    return RUNTIME_DIR


def _code_digest(root: str) -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        try:
            with open(os.path.join(root, *name.split("/")), "rb") as fh:
                h.update(fh.read())
        except OSError:
            # A missing file is itself a distinguishing state: it must not collide with
            # the complete install's address.
            h.update(b"\0missing\0" + name.encode())
    return h.hexdigest()


def socket_path(store_key: str, root: "str | None" = None) -> str:
    """Where the daemon for this store and host, running this code, listens.

    The host arrives inside `store_key`, which is where the rest of the store's identity
    already comes from; this function only truncates. The truncation stays at 16 hex
    characters because macOS caps a unix socket path near 104 bytes, and a readable name
    would not fit.
    """
    here = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    digest = hashlib.sha256(
        f"{store_key}\0{_code_digest(here)}".encode()
    ).hexdigest()[:16]
    # Unix socket paths are length-limited (~104 bytes on macOS), hence the truncation
    # rather than a readable name.
    return os.path.join(runtime_dir(), f"recall-{digest}.sock")


def _host_record():
    """The bound client, imported lazily so `core.host` is not a hard dependency here.

    `lib.ipc` is the module every hook already imports and the one the daemon imports
    first; a top-level import of `core.host` would make the import graph a cycle the day
    anything under `core/` wants an address from here.
    """
    from core.host import active

    return active()


#: Where MCP clients keep the server block we mine for configuration. Checked in order;
#: the first one that names a `memvara` server wins.
#:
#: The paths belong to the client, so they come from its `Host` record. Resolved at import
#: like the rest of this module's addresses, and against `~` rather than `_HOME` because
#: the record states them the way a person would write them down.
_CLIENT_CONFIGS = tuple(
    os.path.expanduser(path) for path in _host_record().client_configs
)


def server_env() -> "dict[str, str]":
    """The `env` block the client launches the memvara MCP server with.

    Lives here rather than in `open.py` because both halves need it and only one of them
    can afford `pathlib`: the client computes the socket address from this before it opens
    anything, so config discovery has to sit on the cheap side of the import graph.

    Empty when no client config names a memvara server. This is discovery, not validation
    — whatever is found goes to `ServerConfig.from_env`, which decides if it is usable.
    """
    for path in _CLIENT_CONFIGS:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            continue
        for name, block in servers.items():
            if "memvara" not in name.lower() or not isinstance(block, dict):
                continue
            env = block.get("env")
            if isinstance(env, dict):
                return {str(k): str(v) for k, v in env.items()}
    return {}


def store_key() -> str:
    """Identity of the store this process would open, without opening it.

    Derived from configuration rather than from a live handle, because the client must
    compute the same address as the daemon *before* paying to open anything.

    The bound host is part of that identity, and not only for tidiness. Six sibling
    repositories vendor these bytes for six different clients, so without it a Codex
    daemon and a Cursor daemon on one machine, over one `MEMVARA_DB`, with a
    byte-identical hook tree, compute one address and the first to bind serves both. The
    host is also what `server_env` reads the env block *out of* -- so two hosts starting
    from the same environment can resolve two different stores, which is the
    store-separation failure again, arriving through a door the rest of this key cannot
    see because it is computed after the host has already chosen where to look.
    """
    env = {**server_env(), **{k: v for k, v in os.environ.items() if k.startswith("MEMVARA_")}}
    db = env.get("MEMVARA_DB") or ""
    if db and db != ":memory:":
        try:
            db = os.path.realpath(os.path.expanduser(db))
        except OSError:
            db = os.path.expanduser(db)
    hosted = ""
    if not db:
        # A hosted install has no MEMVARA_DB, so without this every hosted account on the
        # machine would hash to the same address and share one daemon -- one account's
        # memories answering another's prompts.
        try:
            with open(os.path.join(_HOME, ".memvara", "credentials.json"), encoding="utf-8") as fh:
                creds = json.load(fh)
            hosted = f"{creds.get('server_url','')}|{creds.get('project','')}"
        except (OSError, ValueError):
            hosted = ""

    return "\0".join([
        _host_record().id,
        db,
        hosted,
        env.get("MEMVARA_MODE", ""),
        env.get("MEMVARA_TENANT", ""),
        env.get("MEMVARA_USER", ""),
        env.get("MEMVARA_AGENT", ""),
        env.get("MEMVARA_SESSION", ""),
        env.get("MEMVARA_EMBEDDER", ""),
    ])


def send(path: str, request: dict, timeout: float = CLIENT_TIMEOUT_SEC) -> "str | None":
    """One request, one reply. `None` means "no daemon" — the caller must fall back.

    Every failure collapses to `None` on purpose. A refused connection, a stale socket
    file left by a killed daemon, a hung server, a half-written reply: from the caller's
    side these are one condition, "this path did not work", and the response to all of
    them is the in-process query.
    """
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(path)
        conn.sendall(json.dumps(request).encode("utf-8"))
        # Half-close so the server reads a clean EOF instead of guessing a length.
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", "replace")
    except (OSError, socket.timeout, ValueError):
        return None
    finally:
        try:
            conn.close()
        except OSError:
            pass


# -- hook stdio ---------------------------------------------------------------
#
# These live beside the socket code rather than in `open.py` for one reason: every hook
# needs them, and `open.py` imports `pathlib`. Reaching for `payload()` must not drag a
# 10.5ms import onto a path whose whole budget is ~30ms.


def payload() -> "dict":
    """The hook's stdin JSON, or `{}` when there is nothing readable there."""
    import sys

    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


#: Bounded by truncation rather than rotation, like the capture log beside it: a debugging
#: aid that needs its own maintenance is worse than no aid.
LOG_MAX_BYTES = 64 * 1024


def log_line(name: str, text: str) -> None:
    """Append one line to `~/.memvara/.hooks/<name>.log`, or give up quietly.

    A second logger, deliberately, rather than reusing `lib.write.log`: that module imports
    `pathlib` for the writing hooks that can afford it, and this one is called from the
    per-prompt path where the same import costs 10.5ms measured. `os.path` and a plain
    `open` do the whole job.

    It exists because the write path has had a token ledger since 0.1.2 and the read path
    has had none -- so the hook that spends context on every single prompt was the one
    nobody could measure, which is exactly how it came to spend four times what it needed
    to without anyone noticing.
    """
    import time

    directory = os.path.join(_HOME, ".memvara", ".hooks")
    path = os.path.join(directory, f"{name}.log")
    try:
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > LOG_MAX_BYTES:
            with open(path, "w", encoding="utf-8"):
                pass
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "+00:00"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {text}\n")
    except OSError:
        pass


def plural(n: int, word: str = "memory", many: str = "memories") -> str:
    """`1 memory`, `2 memories`. Shared so the three hooks cannot drift apart on it."""
    return f"{n} {word if n == 1 else many}"


def emit_json(reply: dict) -> None:
    """Print one JSON object: the hook protocol's structured reply.

    Plain stdout from a hook is either context for the model or nothing at all, depending
    on the event, and neither is visible to the person watching the terminal. `systemMessage`
    is, on both of the events this plugin answers, which is the only reason to prefer this
    over `emit`.
    """
    import sys

    sys.stdout.write(json.dumps(reply) + "\n")


def emit(text: str) -> None:
    """Print a block for the model, or print nothing.

    Whitespace-only output is suppressed rather than printed: a lone newline still reads
    as an injected context block to anyone debugging the transcript.
    """
    import sys

    if text and text.strip():
        sys.stdout.write(text.rstrip() + "\n")
