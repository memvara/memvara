"""The two memvara arms, reading from a hosted deployment instead of a local file.

    PYTHONPATH=. python3 demo/harness.py --memory hosted \\
        --hosted-credentials ~/.memvara/demo-credentials.json ...

`demo/baselines.py` builds `memvara` and `memvara_structured` over a local store that
lives and dies inside the process. This module builds the same two arms over the hosted
service, through `memvara.remote` — the client a customer of app.memvara.dev uses — so the
answer-quality run can measure the thing people actually run. Every other arm stays local:
`none`, `full_transcript` and `naive_rag` use no store, and the comparison arms are their
own systems.

## What is the same, and what cannot be

The same: the turns each question may see (`baselines.visible_turns`), the facts the desk
had recorded by then (`baselines.visible_facts`), the retrieval budget (`k` and the
character cap), and the rendering an integration drops into a prompt, which is `recall()`.

Three things are different, and each is stated in the report beside the table rather than
left to be noticed:

* **The vocabulary.** A hosted project's predicate vocabulary is the deployment's, chosen
  by the project's category; a client cannot send `SUPPORT_PREDICATES`. So `plan` and the
  two addresses are not single-valued there, and a new plan would sit beside the old one
  instead of closing it. `apply_facts_hosted` therefore closes a single-valued slot itself
  — `forget(close="ended")` at the instant the new value begins — before writing the new
  value, which is the explicit form of what the declared cardinality does locally.
  `tests/test_demo_hosted.py` checks every slot at every question instant against the
  local structured arm, which has the schema, rather than against expectations of its own.
  The same test found the second consequence: the built-in vocabulary resolves `plan` as
  an alias of `goal`, so on a project whose category brings no support vocabulary the plan
  history is filed as `account.goal`. Reads by slot resolve the alias and the rendered
  block carries each claim's own sentence, so the reader sees the same words; anything
  keyed on the predicate name does not. `folded_predicates()` computes the folds from the
  vocabulary rather than listing them, and the report prints them.
* **Dated reads.** `POST /v1/recall` has no time axis and the client refuses `valid_at`.
  The four questions that carry `about` are read with `search(valid_at=)` instead and
  rendered by `render_dated`, which is the library's own recall renderer called on those
  results — byte-identical to local `recall(valid_at=)` on the same store, which a test
  pins. Nothing here formats a block of its own.
* **Extraction and the episode cap.** The deployment runs its own extractor over the
  turns the `memvara` arm writes, possibly in the background, and its own cap on how many
  turns a read may return; the local arm sets `read_max_episodes=k`, which a hosted client
  cannot. Neither is controlled here, so each context records how many claims its scope
  held at the moment it was read, and the report prints the range.

## Where the writes go, and how often

One scope per run, arm, corpus scale and question instant. Eighteen of the twenty
questions share one `asked_at`, so a run writes three scopes per arm rather than twenty,
and each holds exactly what a question at that instant may see. The scope is a `user`
under the credential's tenant, named by `HostedMemvara.scope_name`.

A scope is written once. `Manifest` records `started` before the first write and
`complete` after the last, in a JSON-lines file kept with the run. A second run with the
same `--hosted-run-id` reads completed scopes without writing them again, which is how the
noise-floor repeat measures the reader twice over the same stored contexts. A scope marked
`started` and never `complete` is a run that died mid-write, and it is refused: replaying
the fact table into a scope that holds part of it would close values at instants they were
never closed at, and there is no safe way to finish it.

## Whose store

Never the one this machine already uses. `load_demo_credential` refuses the default
credentials file, a file holding the same key as it or as `MEMVARA_API_KEY`, and a file
for the same project, before anything is sent. Make a dedicated project in the console,
sign in to it with `memvara login --credentials PATH`, and pass that path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from demo import baselines as bl
from demo.baselines import Context, Question, Turn, Write

from memvara import HashingEmbedder, Memvara, NullLLM
from memvara.retrieve import EpisodeResult
from memvara.schema import PredicateRegistry

__all__ = ["HOSTED_ARMS", "HostedCredential", "HostedMemvara", "Manifest",
           "apply_facts_hosted", "connect", "load_demo_credential", "render_dated"]

#: The arms this module replaces. Every other arm is unchanged by `--memory hosted`.
HOSTED_ARMS = ("memvara", "memvara_structured")

#: Turns per `add` request. Small enough that one failed request loses little and a
#: request body stays well under any proxy's limit; large enough that the scale-10 corpus
#: is a few dozen requests per scope rather than six hundred.
CHUNK = 50

_DEFAULT_SERVER = "https://app.memvara.dev"


# --- the credential ---------------------------------------------------------------


@dataclass(frozen=True)
class HostedCredential:
    """A credential for the demo's own project. `repr` omits the key."""

    api_key: str = field(repr=False)
    base_url: str
    project: str | None
    path: Path


def load_demo_credential(path: str | os.PathLike[str], *,
                         env: Mapping[str, str] | None = None,
                         default_path: str | os.PathLike[str] | None = None,
                         ) -> HostedCredential:
    """The demo's credential from `path`, refused if it could reach this machine's store.

    Three checks, because there are three ways to end up writing a support history into
    the store somebody works in, and each would be invisible until it had happened:

    * `path` *is* the default credentials file;
    * `path` holds the same key as the default file, or as `MEMVARA_API_KEY` — a copy;
    * `path` is for the same project as the default file — a second key minted for it.

    A missing or keyless file is refused too, naming the command that writes one. No
    message quotes a key.
    """
    from memvara.remote.creds import read_credentials_file
    from memvara.server.config import CREDENTIALS_PATH

    environ = os.environ if env is None else env
    default = Path(default_path if default_path is not None else CREDENTIALS_PATH)
    resolved = Path(path).expanduser()
    if resolved.resolve() == default.expanduser().resolve():
        raise SystemExit(
            f"--hosted-credentials: {resolved} is the default credentials file, which "
            "every other hosted caller on this machine reads. The demo writes thousands of "
            "turns and must not share a store with anything. Sign in to a project made "
            "for it with `memvara login --credentials PATH` and pass that PATH.")
    stored = read_credentials_file(resolved)
    if not stored:
        raise SystemExit(
            f"--hosted-credentials: {resolved} holds no credential. Create a project for "
            "the demo in the console, then run `memvara login --credentials "
            f"{resolved}` and choose that project in the browser.")
    home = read_credentials_file(default.expanduser())
    env_key = (environ.get("MEMVARA_API_KEY") or "").strip()
    if stored["api_key"] in {home.get("api_key"), env_key} - {None, ""}:
        raise SystemExit(
            f"--hosted-credentials: {resolved} holds the same key as "
            + ("MEMVARA_API_KEY" if stored["api_key"] == env_key else str(default))
            + ", so it reaches the same store. Use a key minted for the demo's own project.")
    if stored.get("project") and stored.get("project") == home.get("project"):
        raise SystemExit(
            f"--hosted-credentials: {resolved} is for project {stored['project']!r}, the "
            f"same project as {default}. A second key for the same project is the same "
            "store. Create a separate project for the demo.")
    return HostedCredential(
        api_key=stored["api_key"],
        base_url=(stored.get("server_url") or environ.get("MEMVARA_SERVER_URL")
                  or _DEFAULT_SERVER),
        project=stored.get("project"), path=resolved)


def connect(credential: HostedCredential) -> Any:
    """The hosted client for `credential`. Constructing it sends nothing."""
    return Memvara.connect(api_key=credential.api_key, base_url=credential.base_url)


# --- the manifest -----------------------------------------------------------------


class Manifest:
    """Which scopes a run has started and finished writing, one JSON line per event.

    Appended as it happens, like `evalkit.Checkpoint`, so a run killed mid-write leaves
    the evidence that it was: `started` with no `complete`. A torn last line is skipped
    rather than refusing the file, for the same reason the checkpoint skips one.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._status: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._status[str(row["scope"])] = str(row["status"])
                except (ValueError, KeyError, TypeError):
                    continue

    def status(self, scope: str) -> str | None:
        return self._status.get(scope)

    def started(self, scope: str) -> None:
        self._append({"scope": scope, "status": "started"})

    def complete(self, scope: str, **counts: int) -> None:
        self._append({"scope": scope, "status": "complete", **counts})

    def _append(self, row: dict[str, Any]) -> None:
        self._status[row["scope"]] = row["status"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps({**row, "at": datetime.now(timezone.utc).isoformat()})
                      + "\n")


# --- the writes -------------------------------------------------------------------


def _live(scoped: Any, subject: str, predicate: str) -> list[Any]:
    return [c for c in scoped.history(subject, predicate) if c.state == "live"]


def apply_facts_hosted(scoped: Any, facts: Sequence[Write],
                       registry: PredicateRegistry = bl.SUPPORT_REGISTRY) -> None:
    """`baselines.apply_facts` against a hosted scope that cannot be sent the schema.

    The three verbs are the same and mean the same. What changes is who enforces a
    single-valued slot: locally the declared cardinality closes the old value when a new
    one is asserted; here the deployment's vocabulary does not know `plan`, so this
    closes it — on valid time, at the instant the new value begins — whenever `registry`
    says the predicate is single-valued and a different value is standing. An identical
    standing value is left to be reinforced, as the declared cardinality would.

    `remember` is never passed `close`: the hosted method has no such argument and would
    file it under metadata, where it would do nothing and say nothing.
    """
    for fact in facts:
        if fact.mode == "correct":
            standing = _live(scoped, fact.subject, fact.predicate)
            if len(standing) != 1:
                raise ValueError(
                    f"correction of {fact.subject}.{fact.predicate} at {fact.at:%Y-%m-%d} "
                    f"found {len(standing)} standing values, expected exactly 1")
            scoped.supersede(standing[0].id, fact.subject, fact.predicate, fact.obj,
                             at=fact.at, close="retired", valid_from=fact.valid_from,
                             recorded_at=fact.at, text=fact.text,
                             extractor=bl.SUPPORT_EXTRACTOR)
            continue
        if fact.mode == "replace" or registry.functional(fact.predicate):
            standing = _live(scoped, fact.subject, fact.predicate)
            same = fact.mode != "replace" and [c.object for c in standing] == [fact.obj]
            if standing and not same:
                scoped.forget(fact.subject, fact.predicate, at=fact.valid_from,
                              close="ended")
        scoped.remember(fact.subject, fact.predicate, fact.obj,
                        valid_from=fact.valid_from, recorded_at=fact.at, text=fact.text,
                        extractor=bl.SUPPORT_EXTRACTOR)


# --- the dated read ----------------------------------------------------------------

_RENDERER: Memvara | None = None


def render_dated(results: Sequence[Any], valid_at: datetime) -> str:
    """Render `search(valid_at=)` results exactly as `recall(valid_at=)` renders them.

    Calls the library's own renderer — the dated header, the flattening, claims before
    turns, the episode cut — on a store that holds nothing and is used for nothing else,
    because the renderer is a method. The arguments are the ones `Memvara.recall` passes
    on a read with no history, no budget and no ranking, which is the read the local arm
    makes. `tests/test_demo_hosted.py` pins the output byte for byte against
    `recall(valid_at=)` on the same store.
    """
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = Memvara(embedder=HashingEmbedder(dim=8), llm=NullLLM())
    claims = [r for r in results if not isinstance(r, EpisodeResult)]
    kept = [r for r in results
            if isinstance(r, EpisodeResult) and r.explain.selected is True]
    episodes = [r for r in results
                if isinstance(r, EpisodeResult) and r.explain.selected is not True]
    headers = (_RENDERER._recall_header(valid_at), _RENDERER.RECALL_HISTORY_HEADER,
               _RENDERER.RECALL_EPISODE_HEADER)
    return _RENDERER._recall_block(claims, [[] for _ in claims], kept, episodes,
                                   len(claims) + len(kept) + len(episodes), headers)


# --- the arms ----------------------------------------------------------------------


class HostedMemvara:
    """Builds `memvara` and `memvara_structured` against one hosted client.

    `client` is a `RemoteMemvara` (see `connect`). Each arm method has the `Arm`
    signature, so `arms()` drops straight into the harness in place of the local two.
    """

    def __init__(self, client: Any, *, run_id: str, scale: int, manifest: Manifest,
                 facts: Sequence[Write] = bl.SUPPORT_FACTS,
                 registry: PredicateRegistry = bl.SUPPORT_REGISTRY,
                 chunk: int = CHUNK) -> None:
        self.client = client
        self.run_id = run_id
        self.scale = scale
        self.manifest = manifest
        self.facts = tuple(facts)
        self.registry = registry
        self.chunk = chunk

    def scope_name(self, arm: str, question: Question) -> str:
        """The `user` a scope is written under: run, scale, arm and question instant."""
        at = question.asked_at.astimezone(timezone.utc)
        return f"demo-{self.run_id}-s{self.scale}-{arm}-{at:%Y%m%dT%H%M}"

    def arms(self) -> dict[str, Any]:
        return {"memvara": self.memvara, "memvara_structured": self.memvara_structured}

    def _prepared(self, arm: str, question: Question, turns: Sequence[Turn]) -> Any:
        """The scope for this arm and instant, written first if this run has not."""
        name = self.scope_name(arm, question)
        scoped = self.client.scope(user=name)
        status = self.manifest.status(name)
        if status == "complete":
            return scoped
        if status == "started":
            raise SystemExit(
                f"hosted scope {name} was started by an earlier run and never finished "
                f"({self.manifest.path}). Its contents are partial and replaying the facts "
                "into it would close values at the wrong instants. Start again with a new "
                "--hosted-run-id.")
        self.manifest.started(name)
        seen = bl.visible_turns(question, turns)
        for start in range(0, len(seen), self.chunk):
            scoped.add([{"role": t.role, "content": t.text, "ts": t.at}
                        for t in seen[start:start + self.chunk]])
        written = 0
        if arm == "memvara_structured":
            visible = bl.visible_facts(question, self.facts)
            apply_facts_hosted(scoped, visible, self.registry)
            written = len(visible)
        self.manifest.complete(name, turns=len(seen), facts=written)
        return scoped

    def memvara(self, question: Question, turns: Sequence[Turn], *,
                k: int = bl.DEFAULT_K, max_chars: int = bl.MAX_CONTEXT_CHARS) -> Context:
        """The shipped defaults over the service: turns in, `recall()` out."""
        scoped = self._prepared("memvara", question, turns)
        text = bl.clip(scoped.recall(question.text, k=k, include_episodes=True), max_chars)
        return Context(arm="memvara", text=text,
                       turns_visible=len(bl.visible_turns(question, turns)),
                       items_used=bl.count_entries(text), read="recall",
                       claims_in_scope=scoped.count())

    def memvara_structured(self, question: Question, turns: Sequence[Turn], *,
                           k: int = bl.DEFAULT_K,
                           max_chars: int = bl.MAX_CONTEXT_CHARS) -> Context:
        """The integration over the service: turns and the desk's facts in, and a dated
        question read at its own instant — through `search(valid_at=)`, see the module
        docstring."""
        scoped = self._prepared("memvara_structured", question, turns)
        if question.about is None:
            block, read = scoped.recall(question.text, k=k, include_episodes=True), "recall"
        else:
            block = render_dated(scoped.search(question.text, k=k, valid_at=question.about,
                                               include_episodes=True), question.about)
            read = "search"
        text = bl.clip(block, max_chars)
        return Context(arm="memvara_structured", text=text,
                       turns_visible=len(bl.visible_turns(question, turns)),
                       items_used=bl.count_entries(text), read=read,
                       claims_in_scope=scoped.count())

    def backend_note(self, credential: HostedCredential) -> str:
        """The lines the report prints above its tables, naming where these arms read and
        what about that store differs from the local one."""
        folds = ", ".join(f"{name} -> {filed}"
                          for name, filed in sorted(folded_predicates().items()))
        lines = [f"  memory arms: hosted, project {credential.project or '(unnamed)'} on "
                 f"{credential.base_url}, run {self.run_id}, scale {self.scale}",
                 f"  manifest: {self.manifest.path}",
                 "  The hosted project cannot be sent the support schema, so these arms close "
                 "single-valued slots on the client (demo/hosted.py)."]
        if folds:
            lines.append(f"  On the built-in vocabulary these support predicates are filed "
                         f"under another name: {folds}.")
        return "\n".join(lines)


def folded_predicates(registry: PredicateRegistry | None = None,
                      specs: Sequence[Any] = bl.SUPPORT_PREDICATES) -> dict[str, str]:
    """Support predicates a vocabulary files under a different name, as `{name: filed}`.

    Computed from the vocabulary rather than written down, so that the report's line about
    it is true of whatever the built-in vocabulary is on the day the run is made. Today it
    is `{"plan": "goal"}`: `goal` is a declared built-in predicate that lists `plan` as an
    alias. The deployment's vocabulary for a project's category may differ from the
    built-in one; this describes the built-in one, which is what a category with no pack
    of its own gets.
    """
    vocabulary = registry if registry is not None else PredicateRegistry()
    return {spec.name: vocabulary.normalize(spec.name) for spec in specs
            if vocabulary.normalize(spec.name) != spec.name}
