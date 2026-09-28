"""The virtual environments the nightly framework tests run in: one per framework and pin.

memvara has adapters for LangChain, LlamaIndex, CrewAI and LangGraph, and a drop-in for
mem0's API. Its wheel declares a lower bound on each framework: the `langchain`,
`llama-index`, `crewai` and `langgraph` extras, and `mem0ai` in the `bench` extra. The
nightly tier tests each adapter against the real package twice: at that floor, pinned
exactly, and at the newest release pip can install on this interpreter.

Each environment is a virtual environment of its own under CACHE, built with this
interpreter's `venv` and pip. It holds memvara, installed from a wheel built from this
checkout, and the framework at its pin. It is reused from one run to the next while its
key stays the same. The key covers the framework's pinned version, the versions of the
companions installed beside it, the requirement memvara declares on the framework,
memvara's own dependencies and this interpreter, so a change to any of them builds a new
environment, and pruning removes every environment the current pins no longer need.
Everything else the framework depends on is resolved when the environment is built, and
stays at that version until the key changes. memvara itself is reinstalled into a reused
environment on every run, because the code under test changes far more often than
anything in the key.

Everything that runs pip goes through `Pip`, which the fast tests replace with a fake.
Nothing in this module imports a framework.
"""

from __future__ import annotations

import base64
import contextlib
import csv
import email.parser
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from harness.env import REPO, child_env

if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
# scripts/ is not on the path until the lines above put it there.
from nightly.steps import kill_tree, run_command  # noqa: E402

from . import probe

#: Where the environments live. It is read once, when pytest imports this module while it
#: collects, which is before any test points HOME at a temporary folder. So it is under
#: the home of the process that runs the suite: the folder the nightly run keeps from
#: night to night, or the home of whoever runs the nightly tier by hand.
CACHE = Path.home() / ".cache" / "memvara-adversarial" / "frameworks"

#: The file that marks a finished environment. It is written last, so a folder without
#: it is a build that stopped partway.
MARKER = "memvara-environment.json"

#: Raise this by one when the way an environment is built changes, so that every
#: environment built the old way is built again.
RECIPE = 1

#: The two pins each framework is tested at. `pinned()` says what each one asks pip for.
PINS = ("floor", "latest")

#: The most disk the environments may take together: 8 GB, the budget the plan sets.
#: `test_the_environments_fit_the_disk_budget` fails when they take more.
DISK_BUDGET = 8_000_000_000

#: The probe's script, which each environment's own interpreter runs.
PROBE = Path(probe.__file__)

#: Where pip may reach, and nothing else: an address that refuses every connection.
DEAD_PROXY = "http://127.0.0.1:9"

#: The switches each framework documents for turning its telemetry off. llama-index-core
#: sends no telemetry, so it has none.
TELEMETRY_OFF = {
    # LangChain and LangGraph trace to LangSmith only when this is true.
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
    # CrewAI's anonymous telemetry, and its tracing to CrewAI's own platform.
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
    # chromadb, which CrewAI installs.
    "ANONYMIZED_TELEMETRY": "False",
    # mem0.
    "MEM0_TELEMETRY": "False",
    # The Hugging Face hub, which CrewAI installs.
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    # The convention several libraries honour.
    "DO_NOT_TRACK": "1",
}

_PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
            "all_proxy")

#: A variable whose name has one of these parts holds a credential, and a framework that
#: found one could call a paid service with it.
_CREDENTIAL = re.compile(r"(^|_)(API_?KEY|KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)(_|$)",
                         re.IGNORECASE)


@dataclass(frozen=True)
class Framework:
    """One framework memvara has an adapter for."""

    #: The name in test ids and in the checks file's name.
    name: str
    #: The distribution memvara declares a requirement on.
    dist: str
    #: The extra of memvara's wheel that holds that requirement.
    extra: str
    #: Whether memvara is installed with that extra. mem0's requirement lives in the
    #: bench extra, which also pulls in nltk, so memvara goes in without it.
    install_extra: bool = True
    #: Other distributions the checks need. At each pin, pip resolves them beside the
    #: framework, and the environment installs exactly the versions it chose.
    companions: tuple[str, ...] = ()

    @property
    def checks(self) -> Path:
        """The file of checks the probe runs in this framework's environments."""
        return Path(__file__).parent / "nightly" / f"checks_{self.name}.py"


FRAMEWORKS: tuple[Framework, ...] = (
    Framework("langchain", "langchain-core", "langchain"),
    Framework("llamaindex", "llama-index-core", "llama-index"),
    Framework("crewai", "crewai", "crewai"),
    # memvara's requirement is on langgraph-checkpoint, where `BaseStore` lives. The
    # adapter's docstring compiles a graph with the store, and that needs `langgraph`.
    Framework("langgraph", "langgraph-checkpoint", "langgraph", companions=("langgraph",)),
    Framework("mem0", "mem0ai", "bench", install_extra=False),
)


def framework(name: str) -> Framework:
    """The framework with this name."""
    for candidate in FRAMEWORKS:
        if candidate.name == name:
            return candidate
    raise KeyError(name)


class BuildError(RuntimeError):
    """pip, venv or an environment's interpreter failed, or memvara's wheel says something
    this module cannot use. The message ends with the command's own output."""


# -- what memvara declares ---------------------------------------------------------------


@dataclass(frozen=True)
class Declared:
    """What memvara's wheel says it needs."""

    #: memvara's own dependencies: the requirements that belong to no extra.
    dependencies: tuple[str, ...]
    #: Framework name -> the requirement memvara declares on it, such as
    #: "langchain-core>=0.3".
    requirements: dict[str, str]


def declared(metadata: str) -> Declared:
    """Read memvara's dependencies, and each framework's requirement, from the METADATA
    file of memvara's wheel.

    The wheel is what pip reads when it installs `memvara[langchain]`, so it is the
    source to trust, and reading it needs no `tomllib`, which Python 3.10 lacks.
    """
    message = email.parser.Parser().parsestr(metadata)
    dependencies: list[str] = []
    requirements: dict[str, str] = {}
    for line in message.get_all("Requires-Dist") or []:
        requirement = Requirement(line)
        if requirement.marker is None:
            dependencies.append(str(requirement))
            continue
        for candidate in FRAMEWORKS:
            if (canonicalize_name(requirement.name) == canonicalize_name(candidate.dist)
                    and requirement.marker.evaluate({"extra": candidate.extra})):
                requirements[candidate.name] = f"{requirement.name}{requirement.specifier}"
    missing = [f"{f.dist} in the {f.extra} extra" for f in FRAMEWORKS
               if f.name not in requirements]
    if missing:
        raise BuildError("memvara's wheel declares no requirement on " + ", ".join(missing))
    return Declared(tuple(sorted(dependencies)), requirements)


def wheel_metadata(wheel: Path) -> str:
    """The METADATA file inside a wheel."""
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise BuildError(f"{wheel} holds {len(names)} METADATA files, not one")
        return archive.read(names[0]).decode("utf-8")


def floor(requirement: str) -> str:
    """The requirement pinned to the lowest version it admits.

    `langchain-core>=0.3` gives `langchain-core==0.3`, which pip resolves to 0.3.0,
    because a version equals itself with zeros added to the end. A floor that was never
    released therefore fails to install, and the environment's test says so.
    """
    parsed = Requirement(requirement)
    lowest = [spec.version for spec in parsed.specifier if spec.operator in (">=", "==", "~=")]
    if len(lowest) != 1:
        raise ValueError(f"{requirement} names no single lowest version, so it has no "
                         "floor to pin")
    return f"{parsed.name}=={lowest[0]}"


def pinned(pin: str, requirement: str) -> str:
    """The requirement pip is asked to resolve at `pin`. At "floor", that is the
    requirement pinned to the lowest version it admits. At "latest", it is the
    requirement as memvara declares it, which pip resolves to the newest release it can
    install on this interpreter."""
    if pin == "floor":
        return floor(requirement)
    if pin == "latest":
        return requirement
    raise ValueError(f"{pin!r} is not one of the pins {PINS}")


def interpreter() -> str:
    """This interpreter, as one line that changes whenever the interpreter does."""
    base = getattr(sys, "_base_executable", sys.executable)
    return (f"{sys.implementation.name} {platform.python_version()} "
            f"{sysconfig.get_platform()} {base}")


@dataclass(frozen=True)
class Wanted:
    """One environment the current pins need."""

    framework: Framework
    #: The framework's version, as pip resolved it.
    version: str
    #: The requirement memvara declares on the framework, such as "langchain-core>=0.3".
    requirement: str
    #: memvara's own dependencies.
    dependencies: tuple[str, ...]
    #: `interpreter()` of the interpreter the environment is built with.
    interpreter: str
    #: Each of the framework's companions, with the version pip resolved beside it.
    companions: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        """A digest of everything the environment is built from. The pin is not in it, so
        a floor and a newest release that resolve to one version share one environment."""
        body = json.dumps({
            "recipe": RECIPE, "framework": self.framework.name, "dist": self.framework.dist,
            "version": self.version,
            "extra": self.framework.extra if self.framework.install_extra else None,
            "companions": [list(pair) for pair in self.companions],
            "requirement": self.requirement, "dependencies": list(self.dependencies),
            "interpreter": self.interpreter,
        }, sort_keys=True)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @property
    def directory(self) -> str:
        """The folder name of the environment under the cache."""
        return f"{canonicalize_name(self.framework.dist)}-{self.version}-{self.key[:12]}"

    def install_arguments(self, wheel: Path) -> list[str]:
        """What `pip install` is given to build the environment."""
        return [_memvara(self.framework, wheel), f"{self.framework.dist}=={self.version}",
                *(f"{name}=={version}" for name, version in self.companions)]

    def missing_from(self, distributions: Mapping[str, str]) -> list[str]:
        """The pinned releases, of the framework and of each companion, that an
        environment holding `distributions` does not hold. Empty when it holds them all."""
        pins = [(self.framework.dist, self.version), *self.companions]
        return [f"{name}=={version}" for name, version in pins
                if distributions.get(canonicalize_name(name)) != version]


def _memvara(framework: Framework, wheel: Path) -> str:
    """memvara's wheel as `pip install` is given it, with the framework's extra if the
    framework installs it."""
    return f"{wheel}[{framework.extra}]" if framework.install_extra else str(wheel)


def python_in(path: Path) -> Path:
    """The interpreter of the virtual environment at `path`."""
    if os.name == "nt":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def versions_from_report(report: Mapping[str, Any]) -> dict[str, str]:
    """Each distribution's version in the JSON report of `pip install --report`."""
    return {canonicalize_name(item["metadata"]["name"]): item["metadata"]["version"]
            for item in report["install"]}


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _run_in_group(command: Sequence[str | Path], *, timeout: float,
                  **options: Any) -> subprocess.CompletedProcess[bytes]:
    """Run `command` the way `subprocess.run` does, in a process group of its own, with
    its output in pipes. `run_command` in scripts/nightly/steps.py does the same for a
    command whose output goes to a log, and `run_probe` uses that; `Pip._run` needs the
    output itself, so it uses this.

    Past `timeout`, the command and every process it started are stopped, not the
    command alone, and `subprocess.TimeoutExpired` is raised. On POSIX, whatever the
    command left running in its group is stopped when it exits, too. The stopping is
    `kill_tree`, the nightly run's own helper in scripts/nightly/steps.py, so the two
    stop a command the same way, on Windows as well.
    """
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    process: subprocess.Popen[bytes]
    with subprocess.Popen([str(part) for part in command], **options) as process:
        try:
            output, errors = process.communicate(timeout=timeout)
        except BaseException:
            kill_tree(process)
            raise
        if sys.platform != "win32":
            kill_tree(process)
    return subprocess.CompletedProcess(process.args, process.returncode, output, errors)


def _json(text: str, what: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise BuildError(f"{what} printed something that is not JSON: "
                         f"{_tail(text, 5)}") from exc


#: Run with `-I` in an environment: its installed distributions and where they live. The
#: names are normalised as `distributions()` in probe.py normalises them, and as
#: packaging's `canonicalize_name` does, which the suite uses to look them up. The rule is
#: written out because an environment cannot import packaging. Keep all three the same.
_DESCRIBE = (
    "import importlib.metadata, json, re, sysconfig\n"
    "print(json.dumps({'purelib': sysconfig.get_paths()['purelib'], 'distributions': {\n"
    "    re.sub(r'[-_.]+', '-', d.metadata['Name']).lower(): d.version\n"
    "    for d in importlib.metadata.distributions() if d.metadata['Name']}}))\n"
)


class Pip:
    """Runs this interpreter's `venv` and pip, and each environment's own interpreter.

    Every command gets the suite's child environment without `PYTHONPATH`, so pip and the
    environments see only what is installed in them. A command that cannot start, fails,
    runs past its time limit or answers with something unreadable raises `BuildError`,
    with the end of its output where there is one. A command past its time limit is
    stopped together with every process it started.
    """

    def __init__(self, home: Path, *, python: str = sys.executable,
                 timeout: float = 1800.0) -> None:
        self.home = Path(home)
        self.python = python
        self.timeout = timeout

    def _run(self, command: Sequence[str | Path], what: str) -> str:
        self.home.mkdir(parents=True, exist_ok=True)
        env = child_env(self.home, {"PIP_DISABLE_PIP_VERSION_CHECK": "1",
                                    "PIP_NO_INPUT": "1"})
        env.pop("PYTHONPATH", None)
        try:
            done = _run_in_group(command, timeout=self.timeout, env=env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired as exc:
            raise BuildError(f"{what} ran past {self.timeout:.0f} seconds") from exc
        except OSError as exc:
            raise BuildError(f"{what} could not start: {exc}") from exc
        output = done.stdout.decode("utf-8", "replace")
        if done.returncode != 0:
            raise BuildError(f"{what} failed with exit status {done.returncode}:\n"
                             f"{_tail(output + done.stderr.decode('utf-8', 'replace'))}")
        return output

    def wheel(self, source: Path, folder: Path) -> Path:
        """Build memvara's wheel from the checkout at `source` into `folder`."""
        self._run([self.python, "-m", "pip", "wheel", "--quiet", "--no-deps",
                   "--no-cache-dir", "--wheel-dir", folder, source],
                  f"building memvara's wheel from {source}")
        wheels = sorted(Path(folder).glob("memvara-*.whl"))
        if len(wheels) != 1:
            raise BuildError(f"building memvara's wheel left {len(wheels)} wheels in {folder}")
        return wheels[0]

    def resolve(self, requirements: Sequence[str], *, deps: bool = False) -> dict[str, str]:
        """The version pip would install for each requirement on this interpreter. With
        `deps`, pip resolves the requirements' dependencies as well, and the answer names
        every distribution the install would bring."""
        what = "asking pip what " + ", ".join(requirements) + " resolve to"
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            self._run([self.python, "-m", "pip", "install", "--quiet", "--dry-run",
                       *([] if deps else ["--no-deps"]), "--ignore-installed",
                       "--no-cache-dir", "--report", report, *requirements], what)
            return versions_from_report(_json(report.read_text(encoding="utf-8"), what))

    def create(self, path: Path) -> Path:
        """Create a virtual environment at `path`, and return its interpreter."""
        self._run([self.python, "-m", "venv", path],
                  f"creating a virtual environment at {path}")
        return python_in(path)

    def install(self, python: Path, arguments: Sequence[str]) -> None:
        """`pip install` into the environment `python` belongs to."""
        self._run([python, "-m", "pip", "install", "--quiet", "--no-cache-dir", *arguments],
                  "pip install " + " ".join(arguments))

    def describe(self, python: Path) -> tuple[Path, dict[str, str]]:
        """The environment's site-packages folder and its installed distributions."""
        what = f"asking {python} what it has installed"
        body = _json(self._run([python, "-I", "-c", _DESCRIBE], what), what)
        return Path(body["purelib"]), dict(body["distributions"])


# -- one environment ---------------------------------------------------------------------


@dataclass
class Prepared:
    """An environment that is ready, and what it took to make it ready."""

    wanted: Wanted
    path: Path
    python: Path
    #: "built" from nothing, "rebuilt" because the old one was unusable, or "reused".
    how: str
    seconds: float
    purelib: Path
    distributions: dict[str, str]
    #: Bytes on disk.
    size: int


def _read_marker(path: Path) -> dict[str, Any]:
    try:
        found = json.loads((path / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _write_marker(path: Path, body: Mapping[str, Any]) -> None:
    partial = path / f"{MARKER}.partial"
    try:
        partial.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(partial, path / MARKER)
    except OSError as exc:
        raise BuildError(f"writing the marker {path / MARKER} failed: {exc}") from exc


class Cache:
    """The folder of environments, and how one is made ready."""

    def __init__(self, root: Path, pip: Pip, wheel: Path) -> None:
        self.root = Path(root)
        self.pip = pip
        #: memvara's wheel, built from this checkout for this run.
        self.wheel = Path(wheel)

    def prepare(self, wanted: Wanted) -> Prepared:
        """Make the environment ready: reuse it when its key matches, else build it.

        A reused environment gets this checkout's memvara reinstalled. It is built again
        from nothing when that reinstall fails, when its interpreter cannot say what it
        holds, or when it no longer holds its pinned releases, so a damaged cache mends
        itself instead of failing every night. If building it again fails too, the error
        says why it was not reused as well.

        Every failure is a `BuildError`, which the session remembers, including a folder
        that cannot be removed, a marker that cannot be written and a size that cannot be
        measured.
        """
        path = self.root / wanted.directory
        python = python_in(path)
        started = time.monotonic()
        how = "built"
        # Why an environment whose key matched was built again instead of reused.
        not_reused = ""
        if _read_marker(path).get("key") == wanted.key:
            how = "rebuilt"
            try:
                self.pip.install(python, ["--no-deps", "--force-reinstall", str(self.wheel)])
                purelib, distributions = self.pip.describe(python)
            except BuildError as exc:
                not_reused = str(exc)
            else:
                missing = wanted.missing_from(distributions)
                if missing:
                    not_reused = "it no longer holds " + ", ".join(missing)
                else:
                    how = "reused"
        if how != "reused":
            try:
                if path.exists():
                    _remove(path)
                python = self.pip.create(path)
                self.pip.install(python, wanted.install_arguments(self.wheel))
                purelib, distributions = self.pip.describe(python)
            except BuildError as exc:
                if not not_reused:
                    raise
                raise BuildError(f"{exc}\n\nThe environment was being built again because "
                                 f"reusing it failed: {not_reused}") from exc
        seconds = time.monotonic() - started
        if how != "reused":
            _write_marker(path, {
                "key": wanted.key, "framework": wanted.framework.name,
                "dist": wanted.framework.dist, "version": wanted.version,
                "requirement": wanted.requirement, "interpreter": wanted.interpreter,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "build_seconds": round(seconds, 1)})
        return Prepared(wanted, path, python, how, seconds, purelib, distributions,
                        disk_usage(path))

    def prune(self, keep: Iterable[Wanted]) -> list[Path]:
        """Remove every environment in the cache that `keep` does not name.

        Only folders that are environments go: those with a marker, and those a build
        left with a `pyvenv.cfg` but no marker. Anything else in the cache is left alone.
        """
        names = {wanted.directory for wanted in keep}
        removed: list[Path] = []
        if not self.root.is_dir():
            return removed
        for entry in sorted(self.root.iterdir()):
            if entry.name in names or entry.is_symlink() or not entry.is_dir():
                continue
            if (entry / MARKER).exists() or (entry / "pyvenv.cfg").exists():
                shutil.rmtree(entry)
                removed.append(entry)
        return removed


def _remove(path: Path) -> None:
    """Remove an environment's folder, as a `BuildError` when that fails."""
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise BuildError(f"removing the old environment at {path} failed: {exc}") from exc


def _refuse(error: OSError) -> None:
    """Raise a listing error, which `os.walk` would otherwise skip silently."""
    raise error


def disk_usage(path: Path) -> int:
    """Bytes the files under `path` take on disk. A file with several links counts once.

    A folder that cannot be listed, or a file whose size cannot be read, raises
    `BuildError`: a size that silently leaves them out would pass the disk budget."""
    seen: set[tuple[int, int]] = set()
    total = 0
    try:
        for folder, _, files in os.walk(path, onerror=_refuse):
            for name in files:
                info = os.lstat(os.path.join(folder, name))
                if (info.st_dev, info.st_ino) in seen:
                    continue
                seen.add((info.st_dev, info.st_ino))
                blocks = getattr(info, "st_blocks", None)
                total += blocks * 512 if blocks is not None else info.st_size
    except OSError as exc:
        raise BuildError(f"measuring the size of {path} failed: {exc}") from exc
    return total


def over_budget(ready: Iterable[Prepared], budget: int = DISK_BUDGET) -> str | None:
    """None when the environments in `ready` fit in `budget` bytes together. Otherwise a
    message with their total and each one's size, the largest first. An environment that
    two pins share counts once."""
    sizes = {prepared.path: prepared.size for prepared in ready}
    total = sum(sizes.values())
    if total <= budget:
        return None
    lines = [f"the environments take {total / 1e9:.2f} GB together, over the budget of "
             f"{budget / 1e9:.2f} GB:"]
    lines += [f"  {path.name}: {size / 1e6:.0f} MB"
              for path, size in sorted(sizes.items(), key=lambda item: -item[1])]
    return "\n".join(lines)


def stale_files(purelib: Path, checkout: Path = REPO) -> list[str]:
    """The installed memvara files whose content differs from this checkout's.

    pip writes each installed file's hash into the distribution's RECORD, so this needs
    no second copy of the code to compare with. An empty list means the environment runs
    exactly the memvara in `checkout`.
    """
    records = sorted(Path(purelib).glob("memvara-*.dist-info/RECORD"))
    if len(records) != 1:
        return [f"{len(records)} installed copies of memvara, where there should be one"]
    stale: list[str] = []
    checked = 0
    for row in csv.reader(records[0].read_text(encoding="utf-8").splitlines()):
        if len(row) < 2 or not row[1]:
            continue
        # RECORD uses forward slashes, but a tool that wrote it on Windows may not have.
        name = row[0].replace("\\", "/")
        if not name.startswith("memvara/"):
            continue
        algorithm, _, expected = row[1].partition("=")
        source = checkout / name
        actual = ""
        if source.is_file():
            digest = hashlib.new(algorithm, source.read_bytes()).digest()
            actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        checked += 1
        if actual != expected:
            stale.append(name)
    if checked == 0:
        return ["RECORD lists no memvara file"]
    return stale


# -- running the probe -------------------------------------------------------------------


def probe_env(home: Path) -> dict[str, str]:
    """The environment the probe runs in.

    The suite's child environment, without `PYTHONPATH` and without any credential, with
    every framework's telemetry switched off, and with every proxy variable pointing at a
    port that refuses connections. The audit hook in the probe blocks what goes through
    Python's `socket` module; the proxies stop a child process or a library with its own
    networking from reaching further than the machine.
    """
    extra = {**TELEMETRY_OFF, **{name: DEAD_PROXY for name in _PROXIES},
             "NO_PROXY": "", "no_proxy": ""}
    env = child_env(home, extra)
    env.pop("PYTHONPATH", None)
    return {name: value for name, value in env.items() if not _CREDENTIAL.search(name)}


@dataclass
class ProbeRun:
    """One probe run in one environment."""

    run: probe.Run
    #: The probe's exit status, or None when it ran past its time limit and was killed.
    exit_code: int | None
    #: The end of what the probe and the framework printed.
    output: str
    seconds: float


def run_probe(prepared: Prepared, work: Path, *, checks: Path | None = None,
              timeout: float = 900.0) -> ProbeRun:
    """Run the framework's checks in the environment, with the environment's interpreter.

    `checks` defaults to the framework's own checks file. The report, the log of what was
    printed and the probe's home folder are kept in `work`. An interpreter that cannot
    be started raises `BuildError`. A probe past `timeout` is stopped together with every
    process it started, and its exit code is None.
    """
    work.mkdir(parents=True, exist_ok=True)
    name = prepared.wanted.directory
    report = work / f"{name}.jsonl"
    log = work / f"{name}.log"
    home = work / f"{name}-home"
    home.mkdir(exist_ok=True)
    report.unlink(missing_ok=True)
    log.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        done = run_command(
            [str(part) for part in (prepared.python, "-I", "-B", PROBE,
                                    checks or prepared.wanted.framework.checks, report)],
            cwd=work, env=probe_env(home), log=log, deadline=started + timeout)
    except OSError as exc:
        raise BuildError(f"the probe could not start with {prepared.python}: "
                         f"{exc}") from exc
    printed = log.read_text(encoding="utf-8", errors="replace")
    return ProbeRun(probe.read(report), done.returncode, _tail(printed, 60),
                    time.monotonic() - started)


# -- one run of the suite ----------------------------------------------------------------


@dataclass
class Session:
    """One pytest run's environments: what the pins resolved to, and each environment
    once a test has asked for it."""

    cache: Cache
    #: (framework name, pin) -> the environment that pin needs.
    wanted: dict[tuple[str, str], Wanted]
    work: Path
    #: The environments pruning removed when the session began.
    removed: list[Path] = field(default_factory=list)
    _prepared: dict[str, Prepared | BuildError] = field(default_factory=dict)
    _probed: dict[str, ProbeRun | BuildError] = field(default_factory=dict)

    def prepare(self, name: str, pin: str) -> Prepared:
        """The environment for one framework and pin, made ready once per run. A failure
        is remembered, so every test of that environment reports it without retrying."""
        directory = self.wanted[(name, pin)].directory
        if directory not in self._prepared:
            try:
                self._prepared[directory] = self.cache.prepare(self.wanted[(name, pin)])
            except BuildError as exc:
                self._prepared[directory] = exc
        found = self._prepared[directory]
        if isinstance(found, BuildError):
            raise BuildError(str(found))
        return found

    def probe(self, name: str, pin: str) -> ProbeRun:
        """The probe's run in that environment, run once per run."""
        directory = self.wanted[(name, pin)].directory
        if directory not in self._probed:
            try:
                self._probed[directory] = run_probe(self.prepare(name, pin), self.work)
            except BuildError as exc:
                self._probed[directory] = exc
        found = self._probed[directory]
        if isinstance(found, BuildError):
            raise BuildError(str(found))
        return found

    def _ready(self) -> list[tuple[list[str], Prepared]]:
        ready = []
        for directory, prepared in self._prepared.items():
            if isinstance(prepared, Prepared):
                names = [f"{name} {pin}" for (name, pin), wanted in self.wanted.items()
                         if wanted.directory == directory]
                ready.append((names, prepared))
        return ready

    def summary(self) -> list[str]:
        """What each environment resolved to, its size, and how long it took."""
        lines = [f"environments in {self.cache.root}"]
        if self.removed:
            lines.append("removed, because no pin needs them any more: "
                         + ", ".join(path.name for path in self.removed))
        total = 0
        for names, prepared in self._ready():
            framework_ = prepared.wanted.framework
            versions = [f"{framework_.dist} {prepared.wanted.version}"] + [
                f"{companion} {prepared.distributions.get(canonicalize_name(companion), '?')}"
                for companion in framework_.companions]
            probed = self._probed.get(prepared.wanted.directory)
            checks = (f"; the checks took {probed.seconds:.1f} s"
                      if isinstance(probed, ProbeRun) else "")
            lines.append(f"{' and '.join(names)}: {', '.join(versions)}; {prepared.how} in "
                         f"{prepared.seconds:.1f} s; {prepared.size / 1e6:.0f} MB{checks}")
            total += prepared.size
        lines.append(f"{len(self._ready())} environments, {total / 1e9:.2f} GB in all")
        return lines

    def record(self) -> dict[str, Any]:
        """The same, as data, for `last-run.json`."""
        environments = []
        for names, prepared in self._ready():
            probed = self._probed.get(prepared.wanted.directory)
            wanted = prepared.wanted
            environments.append({
                "pins": names, "directory": wanted.directory, "dist": wanted.framework.dist,
                "version": wanted.version, "how": prepared.how,
                "seconds": round(prepared.seconds, 1), "size_bytes": prepared.size,
                "probe_seconds": (round(probed.seconds, 1)
                                  if isinstance(probed, ProbeRun) else None),
                "distributions": {name: prepared.distributions.get(canonicalize_name(name))
                                  for name in ("memvara", wanted.framework.dist,
                                               *wanted.framework.companions)}})
        return {"finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "root": str(self.cache.root), "interpreter": interpreter(),
                "removed": [path.name for path in self.removed],
                "environments": environments}


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on `path`, so that two runs never build or prune the same
    environments at once. POSIX only: the nightly tier runs on macOS and Linux, and on
    Windows the block runs unlocked."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _companions(pip: Pip, wheel: Path, framework: Framework,
                version: str) -> tuple[tuple[str, str], ...]:
    """Each of the framework's companions, with the version pip would install beside
    memvara and the framework at `version`. Pinning it puts it in the environment's key,
    so a new release of a companion builds a new environment."""
    if not framework.companions:
        return ()
    found = pip.resolve([_memvara(framework, wheel), f"{framework.dist}=={version}",
                         *framework.companions], deps=True)
    return tuple((name, found[canonicalize_name(name)]) for name in framework.companions)


@contextlib.contextmanager
def session(root: Path, work: Path, pip: Pip | None = None) -> Iterator[Session]:
    """Resolve every framework's two pins, prune the cache, and hand out environments.

    memvara's wheel is built once, from this checkout, and read for what memvara
    declares. Then pip resolves each floor and each newest release on this interpreter,
    and each companion beside each of them. Every environment no pin needs is removed,
    and the session is handed out. When it
    ends, `root/last-run.json` records what each environment resolved to, its size and
    how long it took.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    work = Path(work)
    with _locked(root / ".lock"):
        pip = pip or Pip(work / "home")
        wheel = pip.wheel(REPO, work / "wheel")
        needs = declared(wheel_metadata(wheel))
        resolved = {pin: pip.resolve([pinned(pin, needs.requirements[f.name])
                                      for f in FRAMEWORKS])
                    for pin in PINS}
        this = interpreter()
        wanted: dict[tuple[str, str], Wanted] = {}
        for f in FRAMEWORKS:
            for pin in PINS:
                version = resolved[pin][canonicalize_name(f.dist)]
                wanted[(f.name, pin)] = Wanted(
                    f, version, needs.requirements[f.name], needs.dependencies, this,
                    _companions(pip, wheel, f, version))
        cache = Cache(root, pip, wheel)
        current = Session(cache, wanted, work, cache.prune(wanted.values()))
        try:
            yield current
        finally:
            (root / "last-run.json").write_text(json.dumps(current.record(), indent=2),
                                                encoding="utf-8")
