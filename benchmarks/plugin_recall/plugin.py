"""Finding a plugin's prompt hook, and running it the way the editor would.

Nothing here knows what a memory is. It knows how Claude Code invokes a `UserPromptSubmit`
hook -- argv, environment, the event on stdin, the reply on stdout -- and that is the whole
vendor-neutral surface this benchmark stands on.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

#: Where Claude Code unpacks installed plugins: `cache/<marketplace>/<plugin>/<version>`.
#: Resolving a bare `--plugin memvara` through this is a convenience; a path always wins,
#: so a checkout under development can be graded without installing it.
CACHE = Path.home() / ".claude" / "plugins" / "cache"

#: The event name being simulated. Only one hook event can put text into the model's
#: context on a per-prompt basis, and this is it.
EVENT = "UserPromptSubmit"


class PluginError(RuntimeError):
    """A plugin could not be resolved or does not declare a prompt hook."""


@dataclass(frozen=True)
class Plugin:
    name: str
    version: str
    root: Path
    argv: list[str]
    timeout: float

    def label(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True)
class Reply:
    """One hook invocation's result.

    `context` is the field this benchmark scores and the field that costs tokens.
    `system_message` is deliberately kept separate and never scored: it renders as a
    status line for the person at the terminal and does not enter the model's context, so
    counting it would charge a plugin for text the model never sees.
    """

    context: str
    system_message: str
    exit_code: int
    elapsed_ms: float
    stdout: str
    stderr: str
    error: str = ""

    @property
    def spoke(self) -> bool:
        return bool(self.context.strip())

    @property
    def tokens(self) -> int:
        # Four characters per token, the same rough divisor the rest of this repository
        # uses for context accounting. Deliberately not a real tokenizer: the figure is
        # compared between plugins measured the same way, and a dependency on one vendor's
        # tokenizer would make the harness harder to run than the thing it grades.
        return round(len(self.context) / 4)


def _plugin_json(root: Path) -> dict:
    path = root / ".claude-plugin" / "plugin.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _resolve_root(target: str) -> Path:
    """A filesystem path, or the newest installed version of a named plugin.

    A bare word is always a plugin name and never a relative directory, which is not
    fussiness: the first run of this harness was inside a repository that has a directory
    named `memvara` at its root, and `--plugin memvara` resolved to the library package
    instead of the installed plugin. It failed loudly there because the package has no
    `hooks/`, but a bare name that happens to match a directory holding some other
    plugin's hooks would have been graded silently as the wrong thing. Anything meant as
    a path says so with a separator -- `./memvara`, `~/x/y`, an absolute path.
    """
    looks_like_path = os.sep in target or target.startswith(("~", "."))
    if looks_like_path:
        candidate = Path(target).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
        raise PluginError(f"No directory at {target!r}.")

    matches = sorted(p for p in CACHE.glob(f"*/{target}/*") if (p / "hooks").is_dir())
    if not matches:
        raise PluginError(
            f"No installed plugin named {target!r}: nothing matches {CACHE}/*/{target}/* "
            "with a hooks/ directory. Install it, or pass a path to a plugin root -- the "
            "directory holding hooks/hooks.json -- written with a separator so it is not "
            "read as a name.")
    # One name can be installed from more than one marketplace -- a published copy and a
    # local development copy, most commonly -- and they are different software with the
    # same name. The first run of this harness picked the wrong one by lexical luck and
    # graded a build that was failing every call. Listing them and stopping costs the
    # caller one flag; guessing costs them a published number about the wrong artefact.
    marketplaces = {path.parent.parent.name for path in matches}
    if len(marketplaces) > 1:
        listed = "\n  ".join(str(path) for path in matches)
        raise PluginError(
            f"{target!r} is installed from {len(marketplaces)} marketplaces and they are "
            f"different software with one name. Pass the path you mean:\n  {listed}")
    # Lexical order over the version segment, within one marketplace. Good enough while
    # versions are zero-padded decimals, and the chosen path is printed in the report, so
    # a wrong pick is visible rather than silent.
    return matches[-1].resolve()


def discover(target: str, *, timeout: float | None = None) -> Plugin:
    """Resolve `target` to a plugin and the argv of its prompt hook.

    Raises rather than degrading. A benchmark that silently grades nothing when it cannot
    find the hook reports a plugin as having said nothing on every prompt, which is a
    perfect score on half the corpus and the single most misleading number this harness
    could produce.
    """
    root = _resolve_root(target)
    hooks_json = root / "hooks" / "hooks.json"
    if not hooks_json.is_file():
        raise PluginError(f"{root} has no hooks/hooks.json, so it declares no prompt hook.")
    try:
        declared = json.loads(hooks_json.read_text())
    except (OSError, ValueError) as exc:
        raise PluginError(f"{hooks_json} is not readable JSON: {exc}") from exc

    entries = [
        hook
        for group in declared.get("hooks", {}).get(EVENT, [])
        for hook in group.get("hooks", [])
        if hook.get("type") == "command" and hook.get("command")
    ]
    if not entries:
        raise PluginError(
            f"{root.name} declares no {EVENT} command hook. It may still be a memory "
            "plugin -- one that recalls through a tool the model chooses to call rather "
            "than through injected context -- but that is a different mechanism and this "
            "harness cannot measure it.")
    if len(entries) > 1:
        raise PluginError(
            f"{root.name} declares {len(entries)} {EVENT} command hooks. Scoring the first "
            "would silently drop the rest; grade a plugin root with exactly one.")

    hook = entries[0]
    # `${CLAUDE_PLUGIN_ROOT}` is the only substitution the editor guarantees, and both
    # plugins read so far use it for every path they name. Substituting before the split
    # rather than after keeps a root containing spaces working, since the quoting in the
    # declared command is what says where the argument boundaries are.
    command = str(hook["command"]).replace("${CLAUDE_PLUGIN_ROOT}", str(root))
    meta = _plugin_json(root)
    return Plugin(
        name=str(meta.get("name") or root.parent.name),
        version=str(meta.get("version") or root.name),
        root=root,
        argv=shlex.split(command),
        timeout=float(timeout if timeout is not None else hook.get("timeout", 10)),
    )


def invoke(plugin: Plugin, prompt: str, *, session_id: str, cwd: Path,
           extra_env: "dict[str, str] | None" = None) -> Reply:
    """Run the hook once, exactly as the editor does, and read its reply.

    A hook that fails is recorded, never raised. The host's own contract is that a hook
    must not fail a prompt, so a crashing hook is a real and reportable outcome -- it said
    nothing, at whatever cost it took to say it -- and turning it into an exception here
    would end the run instead of scoring the plugin honestly.
    """
    event = {
        "session_id": session_id,
        "transcript_path": "",
        "cwd": str(cwd),
        "hook_event_name": EVENT,
        "prompt": prompt,
    }
    # `extra_env` is how a plugin is pointed at a benchmark store instead of the
    # operator's real one -- `MEMVARA_DB`, `SUPERMEMORY_API_URL`, and so on. It is applied
    # last so it wins, and the report prints it, because a result measured against a
    # different store than the reader assumes is the most misleading kind there is.
    env = {**os.environ, "CLAUDE_PLUGIN_ROOT": str(plugin.root),
           "CLAUDE_PROJECT_DIR": str(cwd),
           # The session under test, so a hook can behave differently on a session's
           # first prompt -- which is what a once-per-session preamble does.
           "BENCH_SESSION": session_id, **(extra_env or {})}
    started = time.monotonic()
    try:
        done = subprocess.run(
            plugin.argv, input=json.dumps(event), capture_output=True, text=True,
            timeout=plugin.timeout, cwd=str(cwd), env=env, check=False)
    except subprocess.TimeoutExpired:
        return Reply("", "", -1, (time.monotonic() - started) * 1000, "", "",
                     error=f"timed out after {plugin.timeout}s")
    except OSError as exc:
        return Reply("", "", -1, (time.monotonic() - started) * 1000, "", "", error=str(exc))
    elapsed = (time.monotonic() - started) * 1000

    stdout = done.stdout or ""
    context, system_message = "", ""
    try:
        body = json.loads(stdout)
    except ValueError:
        # Claude Code treats a zero-exit hook's plain stdout as context. A plugin that
        # answers this way is measured on the same terms as one answering in JSON.
        context = stdout if done.returncode == 0 else ""
    else:
        if isinstance(body, dict):
            specific = body.get("hookSpecificOutput") or {}
            context = str(specific.get("additionalContext") or "")
            system_message = str(body.get("systemMessage") or "")
        else:
            context = stdout
    return Reply(context, system_message, done.returncode, elapsed, stdout, done.stderr or "")
