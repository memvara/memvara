"""Per-project extraction guidance: what one project wants remembered, added to the prompt.

The shipped extraction prompt (`base.EXTRACT_SYSTEM`) is written for every user of the
library, so it cannot know that one project cares about deployment decisions and another
never wants a stack trace stored. `Guidance` carries that knowledge: a short description
of the project and two lists of rules, one for what to extract and one for what to leave
out. It is **appended** to the system message under a fixed heading. It never replaces
the shipped rules, which is the job of `MEMVARA_LLM_EXTRACT_SYSTEM` (a full replacement,
and a different decision).

The guidance goes in the system message and nowhere else, because it is an instruction.
The conversation turns go in the user message as data. Putting operator rules beside the
content would let a turn that quotes rules pass itself off as one.

`with_guidance` is the one function that builds the combined system message. Both model
backends call it, and so can any other extractor that sends its own system message, so
every path adds the guidance under the same heading in the same words.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = ["GUIDANCE_HEADING", "Guidance", "GuidanceError", "MAX_CONTEXT_CHARS",
           "MAX_RULE_CHARS", "MAX_RULES", "load_guidance", "with_guidance"]

#: Most characters `Guidance.context` may hold. A paragraph or two describing a project.
MAX_CONTEXT_CHARS = 1500

#: Most rules `Guidance.include` may hold, and separately most `Guidance.exclude` may hold.
MAX_RULES = 20

#: Most characters one rule may hold. A rule is one sentence.
MAX_RULE_CHARS = 200

#: The line the guidance starts under. Fixed, so a reader of a request log can find where
#: the shipped prompt ends and the project's own rules begin.
GUIDANCE_HEADING = "Project guidance from the operator of this memory store"

#: The sentence under the heading. It says the guidance narrows what is worth extracting
#: and does not change the rules above it, so a rule such as "store everything" cannot be
#: read as permission to break the output format.
_PREAMBLE = (
    "The operator of this memory store wrote the guidance below for this project. Use it "
    "to decide which facts are worth extracting. It does not change the output format or "
    "any rule above.")

#: The three field names a guidance file may use, and nothing else.
_FIELDS = ("context", "include", "exclude")

#: Most bytes a guidance file may hold. The longest valid guidance is about 10 KB, so this
#: catches a path pointing at the wrong file (a log, a model) before it is read whole.
_MAX_FILE_BYTES = 64 * 1024


class GuidanceError(ValueError):
    """The guidance is too long, has the wrong type, or its file cannot be read."""


@dataclass(frozen=True, slots=True)
class Guidance:
    """What one project wants extracted, appended to the shipped extraction prompt.

    `context` describes the project in a paragraph (at most `MAX_CONTEXT_CHARS`
    characters). `include` lists things worth extracting and `exclude` lists things to
    leave out; each list holds at most `MAX_RULES` rules of at most `MAX_RULE_CHARS`
    characters. Anything longer is refused with `GuidanceError` rather than cut, because a
    rule cut in half can say the opposite of what its author wrote.

    Rules are stripped of surrounding whitespace, and a blank rule is refused. Lists are
    stored as tuples, so a `Guidance` can be shared between threads and used as a
    dictionary key.

    >>> g = Guidance(context="A payments service.", include=["decisions about retries"])
    >>> g.include
    ('decisions about retries',)
    >>> Guidance().is_empty
    True
    >>> Guidance(exclude=[""])
    Traceback (most recent call last):
    ...
    memvara.llm.guidance.GuidanceError: guidance exclude rule 1 is blank. Remove it, or write the rule.
    """

    context: str = ""
    include: Sequence[str] = ()
    exclude: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.context, str):
            raise GuidanceError(
                f"guidance context must be text, not {type(self.context).__name__}.")
        context = self.context.strip()
        if len(context) > MAX_CONTEXT_CHARS:
            raise GuidanceError(
                f"guidance context is {len(context)} characters, over the "
                f"{MAX_CONTEXT_CHARS} allowed. Describe the project in a paragraph or two; "
                "put anything rule-shaped in include or exclude.")
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "include", _rules(self.include, "include"))
        object.__setattr__(self, "exclude", _rules(self.exclude, "exclude"))

    @property
    def is_empty(self) -> bool:
        """True when there is nothing to add to the prompt."""
        return not (self.context or self.include or self.exclude)

    def render(self) -> str:
        """The text appended to a system message, heading included, or "" when empty.

        >>> print(Guidance(context="A CLI tool.", exclude=["stack traces"]).render())
        Project guidance from the operator of this memory store:
        The operator of this memory store wrote the guidance below for this project. Use it to decide which facts are worth extracting. It does not change the output format or any rule above.
        About this project: A CLI tool.
        Do not extract:
        - stack traces
        """
        if self.is_empty:
            return ""
        lines = [f"{GUIDANCE_HEADING}:", _PREAMBLE]
        if self.context:
            lines.append(f"About this project: {self.context}")
        if self.include:
            lines.append("Extract:")
            lines.extend(f"- {rule}" for rule in self.include)
        if self.exclude:
            lines.append("Do not extract:")
            lines.extend(f"- {rule}" for rule in self.exclude)
        return "\n".join(lines)


def _rules(value: Any, field: str) -> tuple[str, ...]:
    """One list of rules, checked and stripped. See `Guidance`."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        # A bare string is refused rather than read as one rule, because iterating it
        # would make every character a rule.
        raise GuidanceError(
            f"guidance {field} must be a list of rules, not {type(value).__name__}.")
    if len(value) > MAX_RULES:
        raise GuidanceError(
            f"guidance {field} has {len(value)} rules, over the {MAX_RULES} allowed. "
            "Merge rules that say the same thing.")
    rules = []
    for position, rule in enumerate(value, start=1):
        if not isinstance(rule, str):
            raise GuidanceError(
                f"guidance {field} rule {position} must be text, not "
                f"{type(rule).__name__}.")
        text = rule.strip()
        if not text:
            raise GuidanceError(
                f"guidance {field} rule {position} is blank. Remove it, or write the rule.")
        if len(text) > MAX_RULE_CHARS:
            raise GuidanceError(
                f"guidance {field} rule {position} is {len(text)} characters, over the "
                f"{MAX_RULE_CHARS} allowed. A rule is one sentence.")
        rules.append(text)
    return tuple(rules)


def with_guidance(system: str, guidance: Guidance | None) -> str:
    """`system` with the guidance appended under its heading, or `system` unchanged.

    The function every extractor uses to add guidance to its own system message, so the
    heading and wording are the same whichever path makes the model call. `None` and an
    empty `Guidance` both return `system` exactly, so a caller with no guidance sends the
    request it sent before guidance existed, byte for byte.

    >>> with_guidance("Extract facts.", None)
    'Extract facts.'
    >>> with_guidance("Extract facts.", Guidance(include=["deadlines"])).splitlines()[:3]
    ['Extract facts.', '', 'Project guidance from the operator of this memory store:']
    """
    if guidance is None or guidance.is_empty:
        return system
    return f"{system}\n\n{guidance.render()}"


def load_guidance(path: str | Path) -> Guidance:
    """Read a guidance file: TOML with `context`, `include` and `exclude`, all optional.

    ```toml
    context = "A payments service. The team cares about incidents and why they happened."
    include = ["decisions about retries and timeouts", "who owns which service"]
    exclude = ["stack traces", "anything about the weekend"]
    ```

    An unknown key is refused rather than ignored: `exlude = [...]` would otherwise leave
    the rule out of every extraction with nothing saying why. The file is read once, when
    the server starts.

    **This needs Python 3.11 or later**, where the standard library's TOML reader
    (`tomllib`) arrives. On Python 3.10 this raises `GuidanceError` saying so, and
    everything else in memvara keeps working. That follows the decision
    `schema._toml_reader` records for predicate vocabularies: no `tomli` fallback, because
    it would be a runtime dependency the package does not declare, and no second
    hand-written reader to keep in step with the real one. A caller on 3.10 can still
    build a `Guidance` in Python and pass it to `Memvara(extract_guidance=...)`.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
        raise GuidanceError(
            "An extraction guidance file needs Python 3.11 or later, which is where "
            "`tomllib` arrives. Everything else in memvara works on 3.10; there, build a "
            "Guidance in Python and pass it to Memvara(extract_guidance=...).") from None
    try:
        size = os.path.getsize(path)
        if size > _MAX_FILE_BYTES:
            raise GuidanceError(
                f"guidance file {str(path)!r} is {size} bytes, over the {_MAX_FILE_BYTES} "
                "this reads. Guidance is a few kilobytes, so the path points at another "
                "file.")
        with open(path, "rb") as handle:
            body = tomllib.load(handle)
    except OSError as exc:
        raise GuidanceError(f"guidance file {str(path)!r} cannot be read: {exc}.") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise GuidanceError(f"guidance file {str(path)!r} is not valid TOML: {exc}.") \
            from None
    unknown = sorted(set(body) - set(_FIELDS))
    if unknown:
        raise GuidanceError(
            f"guidance file {str(path)!r} has unknown key(s) {', '.join(unknown)}. The keys "
            "are context, include and exclude.")
    try:
        return Guidance(**body)
    except GuidanceError as exc:
        raise GuidanceError(f"guidance file {str(path)!r}: {exc}") from None
