"""Findings: the record a layer of the suite writes when it sees a break.

A finding is one JSON line. The nightly run (scripts/nightly/run.py) turns each failed
test into a finding, reads the findings its steps write, and keeps them all in the night's
findings.jsonl. `Finding.signature()` is the fingerprint the run deduplicates by: a break
seen on two nights has one fingerprint, so it becomes one issue, not two.

The signature covers the fields that identify a break: the layer that found it, the
surface it was seen through, the invariant that failed, and the operations that replay
it. It leaves out the fields that change from one night to the next (the seed, the commit
and the paths of the artifacts), the severity, which triage can change, and the title and
the failure text, which often hold temporary paths and timings. So the producer must give
the operations in a minimal, deterministic form, with no temporary paths or wall-clock
times in them, or the same break gets a new fingerprint every night.

>>> first = Finding("model", "library", "reads match the model",
...                 ops=["Forget(user='u1')"], seed="one seed", commit="aaaa")
>>> second = Finding("model", "library", "reads match the model",
...                  ops=["Forget(user='u1')"], seed="another seed", commit="bbbb")
>>> first.signature() == second.signature()
True
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping, Sequence

#: What a finding can be, which decides where the nightly run may file it.
#:
#: * ``unclassified``: nobody has checked it against the "In scope" section of
#:   SECURITY.md yet. The nightly run never files an unclassified finding in public.
#: * ``security``: it is in scope for SECURITY.md. It goes to a private draft advisory,
#:   never to a public issue, branch or pull request.
#: * ``data-loss``, ``wrong-result`` and ``crash``: someone checked, found it out of scope
#:   for SECURITY.md, and it can be filed as a public issue with a strict-xfail test.
SEVERITIES = ("unclassified", "security", "data-loss", "wrong-result", "crash")

#: Part of every signature. Changing how signatures are computed changes every
#: fingerprint, so every open break would be filed again: raise this only on purpose.
SIGNATURE_VERSION = 1

#: The fields that identify a finding; `signature()` hashes these and nothing else.
_IDENTITY = ("layer", "surface", "invariant")


@dataclass(frozen=True)
class Finding:
    """One break, as a layer of the suite saw it."""

    #: The part of the suite that found it, such as ``model`` or ``redteam``.
    layer: str
    #: What it was seen through, such as ``library``, ``server`` or ``hooks``.
    surface: str
    #: The property that failed: an invariant's name, or a test's node id.
    invariant: str
    #: One of SEVERITIES.
    severity: str = "unclassified"
    #: The operations that replay the break, as JSON values, in a minimal form.
    ops: Sequence[Any] = ()
    #: What reproduces a random search, such as Hypothesis's ``@reproduce_failure`` call.
    seed: str | None = None
    #: Files kept with the finding, by name, as paths relative to the night's folder.
    artifacts: Mapping[str, str] = field(default_factory=dict)
    #: The commit that was tested.
    commit: str = ""
    #: One line for a person, used as an issue's title.
    title: str = ""
    #: The failure's text, for the report and an issue's body.
    detail: str = ""

    def __post_init__(self) -> None:
        for name in _IDENTITY:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"a finding needs a non-empty {name}, not {value!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}; a finding's severity "
                             f"is one of {', '.join(SEVERITIES)}")
        ops = tuple(self.ops)
        try:
            json.dumps(list(ops), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"a finding's ops must be JSON values: {exc}") from None
        object.__setattr__(self, "ops", ops)
        object.__setattr__(self, "artifacts",
                           {str(name): str(path) for name, path in
                            sorted(dict(self.artifacts).items())})

    def __hash__(self) -> int:
        return hash(self.signature())

    def signature(self) -> str:
        """The fingerprint that deduplicates this break across nights: the SHA-256 of the
        identity fields and the ops, as JSON with sorted keys, no spaces, and text as UTF-8
        rather than escapes."""
        payload = json.dumps(
            {"v": SIGNATURE_VERSION, "layer": self.layer, "surface": self.surface,
             "invariant": self.invariant, "ops": list(self.ops)},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Every field, plus the fingerprint, so a reader can search a file for it."""
        return {"layer": self.layer, "surface": self.surface, "invariant": self.invariant,
                "severity": self.severity, "ops": list(self.ops), "seed": self.seed,
                "artifacts": dict(self.artifacts), "commit": self.commit,
                "title": self.title, "detail": self.detail,
                "fingerprint": self.signature()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Finding:
        """The finding `data` describes. An unknown key is refused, because a misspelt
        field would otherwise be lost without a word. A ``fingerprint`` key must equal the
        signature, or the finding was written by a run that computed signatures
        differently."""
        known = {each.name for each in fields(cls)}
        unknown = sorted(set(data) - known - {"fingerprint"})
        if unknown:
            raise ValueError(f"unknown finding field: {', '.join(unknown)}")
        missing = [name for name in _IDENTITY if name not in data]
        if missing:
            raise ValueError(f"a finding needs {', '.join(missing)}")
        finding = cls(**{key: value for key, value in data.items() if key in known})
        stored = data.get("fingerprint")
        if stored is not None and stored != finding.signature():
            raise ValueError(f"the stored fingerprint {stored} is not this finding's "
                             f"signature {finding.signature()}: it was computed differently")
        return finding

    def to_json(self) -> str:
        """One line of JSON. Text outside ASCII is escaped, so no character in a finding
        can split its line, not even a Unicode line separator."""
        return json.dumps(self.to_dict(), ensure_ascii=True, allow_nan=False)

    @classmethod
    def from_json(cls, line: str) -> Finding:
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"a finding is a JSON object, not {type(data).__name__}")
        return cls.from_dict(data)


def write(path: pathlib.Path | str, findings: Iterable[Finding]) -> None:
    """Replace the file at `path` with `findings`, one line each."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for finding in findings:
            handle.write(finding.to_json() + "\n")


def append(path: pathlib.Path | str, finding: Finding) -> None:
    """Add one finding to the end of the file at `path`."""
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(finding.to_json() + "\n")


def read(path: pathlib.Path | str) -> list[Finding]:
    """The findings in the file at `path`, in order. Blank lines are ignored; a line that
    is not a finding raises ValueError naming its line number."""
    findings = []
    text = pathlib.Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            findings.append(Finding.from_json(line))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}, line {number}: {exc}") from None
    return findings
