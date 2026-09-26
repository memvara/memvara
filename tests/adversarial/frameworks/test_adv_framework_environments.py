"""The virtual environments the nightly framework tests run in: how each is built,
reused, rebuilt and removed.

pip is replaced here by `FakePip`, which makes folders and small files where pip would
download and install, so these tests need no network and no framework. The nightly tier
runs the same code with the real pip.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from harness.env import REPO
from harness.hooks import process_alive

from . import environments, probe

#: The lines hatchling writes into memvara's METADATA, with an extra no framework uses.
METADATA = textwrap.dedent("""\
    Metadata-Version: 2.4
    Name: memvara
    Version: 0.0.1
    Requires-Dist: numpy>=1.24
    Provides-Extra: bench
    Requires-Dist: mem0ai>=2.0; extra == 'bench'
    Requires-Dist: nltk>=3.8; extra == 'bench'
    Provides-Extra: cloud
    Requires-Dist: httpx>=0.27; extra == 'cloud'
    Provides-Extra: crewai
    Requires-Dist: crewai>=1.10.1; extra == 'crewai'
    Provides-Extra: langchain
    Requires-Dist: langchain-core>=0.3; extra == 'langchain'
    Provides-Extra: langgraph
    Requires-Dist: langgraph-checkpoint>=4.1; extra == 'langgraph'
    Provides-Extra: llama-index
    Requires-Dist: llama-index-core>=0.13; extra == 'llama-index'
""")

#: What FakePip says the newest release of each framework, and of each companion, is.
LATEST = {"langchain-core": "1.6.5", "llama-index-core": "0.14.25", "crewai": "1.15.22",
          "langgraph-checkpoint": "4.2.0", "mem0ai": "2.2.1", "langgraph": "1.2.12"}


class FakePip:
    """Stands in for `environments.Pip`, touching nothing but temporary folders.

    It records each call as a tuple whose first item is its kind: "wheel", "resolve",
    "resolve with dependencies", "create", "install", "refresh" (an install with
    --force-reinstall) or "describe". A kind in `failing` raises `BuildError` instead,
    every time, and a kind in `failing_once` raises it the next time only. With
    `blocks_marker`, an install leaves a folder where the marker's temporary file goes,
    so that writing the marker fails the way it does on a full disk.
    """

    def __init__(self, latest: Mapping[str, str] | None = None) -> None:
        self.latest = dict(LATEST if latest is None else latest)
        self.calls: list[tuple[str, ...]] = []
        self.failing: set[str] = set()
        self.failing_once: set[str] = set()
        self.blocks_marker = False

    def _call(self, kind: str, *details: str) -> None:
        self.calls.append((kind, *details))
        if kind in self.failing_once:
            self.failing_once.discard(kind)
            raise environments.BuildError(f"planted failure of {kind}")
        if kind in self.failing:
            raise environments.BuildError(f"planted failure of {kind}")

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]

    def wheel(self, source: Path, folder: Path) -> Path:
        self._call("wheel", str(source))
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "memvara-0.0.1-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("memvara-0.0.1.dist-info/METADATA", METADATA)
        return path

    def resolve(self, requirements: Sequence[str], *, deps: bool = False) -> dict[str, str]:
        self._call("resolve with dependencies" if deps else "resolve", *requirements)
        found: dict[str, str] = {}
        for text in requirements:
            if ".whl" in text:
                continue
            requirement = Requirement(text)
            name = canonicalize_name(requirement.name)
            pinned = [spec.version for spec in requirement.specifier if spec.operator == "=="]
            found[name] = pinned[0] if pinned else self.latest[name]
        return found

    def create(self, path: Path) -> Path:
        self._call("create", path.name)
        assert not path.exists(), "a build must start from an empty folder"
        path.mkdir(parents=True)
        (path / "pyvenv.cfg").write_text("home = /nowhere\n", encoding="utf-8")
        python = environments.python_in(path)
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        return python

    def install(self, python: Path, arguments: Sequence[str]) -> None:
        self._call("refresh" if "--force-reinstall" in arguments else "install", *arguments)
        site = _site(python)
        site.mkdir(parents=True, exist_ok=True)
        if self.blocks_marker:
            (python.parent.parent / f"{environments.MARKER}.partial").mkdir(exist_ok=True)
        for argument in arguments:
            if "==" in argument:
                name, version = argument.split("==")
                (site / f"{canonicalize_name(name)}=={version}").write_text("")
            elif ".whl" in argument:
                (site / "memvara==0.0.1").write_text("")

    def describe(self, python: Path) -> tuple[Path, dict[str, str]]:
        self._call("describe")
        site = _site(python)
        return site, dict(item.name.split("==") for item in site.glob("*==*"))


def _site(python: Path) -> Path:
    return python.parent.parent / "site-packages"


def _wanted(**changes: object) -> environments.Wanted:
    values: dict[str, object] = {
        "framework": environments.framework("langchain"), "version": "0.3.0",
        "requirement": "langchain-core>=0.3", "dependencies": ("numpy>=1.24",),
        "interpreter": "cpython 3.13.14 macosx-11.0-arm64 /usr/local/bin/python3.13"}
    values.update(changes)
    return environments.Wanted(**values)  # type: ignore[arg-type]


def _langgraph(companion: str = "1.2.12") -> environments.Wanted:
    return _wanted(framework=environments.framework("langgraph"), version="4.1.0",
                   requirement="langgraph-checkpoint>=4.1",
                   companions=(("langgraph", companion),))


def _cache(tmp_path: Path) -> tuple[FakePip, environments.Cache]:
    pip = FakePip()
    wheel = pip.wheel(REPO, tmp_path / "wheel")
    pip.calls.clear()
    return pip, environments.Cache(tmp_path / "cache", pip, wheel)  # type: ignore[arg-type]


def _alive(pid: int) -> bool:
    """Whether process `pid` still runs. On Windows the kernel is asked for the process's
    exit code, because os.kill with signal 0 would end the process there."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    return process_alive(pid)


def _poll(condition: Callable[[], bool], seconds: float) -> bool:
    """Whether `condition` becomes true within `seconds`, checking every 50 ms."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def _stop(pid: int) -> None:
    """End process `pid` if it still runs, so that a failing test leaves nothing behind."""
    if _alive(pid):
        try:
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass


#: Starts a process of its own, writes that process's id to a file, and hangs. Formatted
#: with the file's path.
STARTS_A_PROCESS_AND_HANGS = (
    "import subprocess, sys, time\n"
    "from pathlib import Path\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
    "Path({pid_file!r}).write_text(str(child.pid))\n"
    "time.sleep(300)\n")


def _record_row(path: str, content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    return f"{path},sha256={digest.decode('ascii')},{len(content)}"


# -- what the pins are -------------------------------------------------------------------


def test_the_floor_is_the_lowest_version_a_requirement_admits() -> None:
    assert environments.floor("langchain-core>=0.3") == "langchain-core==0.3"
    assert environments.floor("crewai>=1.10.1,<2") == "crewai==1.10.1"
    assert environments.floor("mem0ai==2.0") == "mem0ai==2.0"


@pytest.mark.parametrize("requirement", ["crewai<2", "crewai"])
def test_a_requirement_with_no_lower_bound_has_no_floor(requirement: str) -> None:
    with pytest.raises(ValueError, match="no single lowest version"):
        environments.floor(requirement)


def test_the_wheel_gives_each_framework_its_requirement_and_memvara_its_dependencies() -> None:
    declared = environments.declared(METADATA)
    assert declared.dependencies == ("numpy>=1.24",)
    assert declared.requirements == {
        "langchain": "langchain-core>=0.3", "llamaindex": "llama-index-core>=0.13",
        "crewai": "crewai>=1.10.1", "langgraph": "langgraph-checkpoint>=4.1",
        "mem0": "mem0ai>=2.0"}


def test_a_framework_the_wheel_does_not_name_is_an_error() -> None:
    without = "\n".join(line for line in METADATA.splitlines() if "crewai" not in line)
    with pytest.raises(environments.BuildError, match="crewai in the crewai extra"):
        environments.declared(without)


def test_pips_report_gives_one_version_per_distribution() -> None:
    report = {"version": "1", "install": [
        {"metadata": {"name": "LangChain_Core", "version": "1.6.5"}, "requested": True},
        {"metadata": {"name": "mem0ai", "version": "2.2.1"}, "requested": True}]}
    assert environments.versions_from_report(report) == {"langchain-core": "1.6.5",
                                                         "mem0ai": "2.2.1"}


def test_the_key_changes_with_everything_the_environment_depends_on() -> None:
    """The key decides reuse, so each input an environment is built from must move it.
    The pin is not an input: the floor and the newest release at one version are one
    environment."""
    base = _wanted()
    for change in ({"version": "0.3.1"}, {"requirement": "langchain-core>=0.3.1"},
                   {"dependencies": ("numpy>=2",)},
                   {"interpreter": "cpython 3.13.15 macosx-11.0-arm64 /usr/local/bin/python3.13"},
                   {"framework": environments.framework("langgraph")}):
        assert _wanted(**change).key != base.key, change
    assert _wanted().directory == base.directory
    assert base.directory == f"langchain-core-0.3.0-{base.key[:12]}"
    # A new release of a companion builds a new environment, rather than leaving the
    # release the environment was first built with in place for good.
    assert _langgraph("1.2.13").key != _langgraph("1.2.12").key


def test_the_install_arguments_pin_the_framework_beside_memvaras_own_extra() -> None:
    wheel = Path("memvara-0.0.1-py3-none-any.whl")
    assert _wanted().install_arguments(wheel) == [f"{wheel}[langchain]",
                                                  "langchain-core==0.3.0"]
    assert _langgraph().install_arguments(wheel) == [
        f"{wheel}[langgraph]", "langgraph-checkpoint==4.1.0", "langgraph==1.2.12"]
    mem0 = _wanted(framework=environments.framework("mem0"), version="2.0.0")
    assert mem0.install_arguments(wheel) == [str(wheel), "mem0ai==2.0.0"]


# -- building and reusing ----------------------------------------------------------------


def test_a_missing_environment_is_built_and_then_reused(tmp_path: Path) -> None:
    pip, cache = _cache(tmp_path)
    wanted = _wanted()
    first = cache.prepare(wanted)
    assert first.how == "built" and first.distributions["langchain-core"] == "0.3.0"
    assert pip.kinds() == ["create", "install", "describe"]
    marker = json.loads((first.path / environments.MARKER).read_text(encoding="utf-8"))
    assert marker["key"] == wanted.key
    pip.calls.clear()
    again = cache.prepare(wanted)
    assert again.how == "reused" and again.path == first.path
    assert pip.calls == [("refresh", "--no-deps", "--force-reinstall", str(cache.wheel)),
                         ("describe",)]


def test_a_folder_left_by_an_interrupted_build_is_built_again(tmp_path: Path) -> None:
    """A folder with no marker is a build that stopped partway, and it is removed
    before the build starts: FakePip.create refuses a folder that exists."""
    pip, cache = _cache(tmp_path)
    wanted = _wanted()
    half = cache.root / wanted.directory
    half.mkdir(parents=True)
    (half / "pyvenv.cfg").write_text("", encoding="utf-8")
    assert cache.prepare(wanted).how == "built"
    assert pip.kinds() == ["create", "install", "describe"]


def test_an_environment_whose_refresh_fails_is_built_again(tmp_path: Path) -> None:
    pip, cache = _cache(tmp_path)
    cache.prepare(_wanted())
    pip.failing.add("refresh")
    assert cache.prepare(_wanted()).how == "rebuilt"


def test_an_environment_without_its_pinned_release_is_built_again(tmp_path: Path) -> None:
    pip, cache = _cache(tmp_path)
    prepared = cache.prepare(_wanted())
    for installed in prepared.purelib.glob("langchain-core==*"):
        installed.unlink()
    rebuilt = cache.prepare(_wanted())
    assert rebuilt.how == "rebuilt" and rebuilt.distributions["langchain-core"] == "0.3.0"


def test_an_environment_without_its_pinned_companion_is_built_again(tmp_path: Path) -> None:
    pip, cache = _cache(tmp_path)
    prepared = cache.prepare(_langgraph())
    assert prepared.distributions["langgraph"] == "1.2.12"
    (prepared.purelib / "langgraph==1.2.12").unlink()
    rebuilt = cache.prepare(_langgraph())
    assert rebuilt.how == "rebuilt" and rebuilt.distributions["langgraph"] == "1.2.12"


def test_an_environment_that_cannot_say_what_it_holds_is_built_again(tmp_path: Path) -> None:
    """The reinstall of memvara worked, but asking the environment's interpreter what it
    holds failed. The environment cannot be trusted, so it is built again rather than
    failing every test of it, night after night."""
    pip, cache = _cache(tmp_path)
    cache.prepare(_wanted())
    pip.failing_once.add("describe")
    rebuilt = cache.prepare(_wanted())
    assert rebuilt.how == "rebuilt" and rebuilt.distributions["langchain-core"] == "0.3.0"


def test_a_failed_rebuild_also_says_why_the_environment_was_not_reused(
        tmp_path: Path) -> None:
    """When reusing an environment fails and building it again fails too, the error keeps
    both failures. The first one is often the one that explains the second."""
    pip, cache = _cache(tmp_path)
    prepared = cache.prepare(_wanted())
    pip.failing_once.add("refresh")
    pip.failing.add("install")
    with pytest.raises(environments.BuildError) as raised:
        cache.prepare(_wanted())
    assert "planted failure of install" in str(raised.value)
    assert "planted failure of refresh" in str(raised.value)
    pip.failing.clear()
    cache.prepare(_wanted())
    for installed in prepared.purelib.glob("langchain-core==*"):
        installed.unlink()
    pip.failing.add("install")
    with pytest.raises(environments.BuildError) as raised:
        cache.prepare(_wanted())
    assert "no longer holds langchain-core==0.3.0" in str(raised.value)


def test_a_marker_that_cannot_be_written_is_a_build_error(tmp_path: Path) -> None:
    """A full disk or a folder without write permission stops the marker from being
    written. That must be a build error, which the session remembers, and not an OSError
    that every test of the environment meets again by building it again."""
    pip, cache = _cache(tmp_path)
    pip.blocks_marker = True
    with pytest.raises(environments.BuildError, match="writing the marker"):
        cache.prepare(_wanted())


def test_a_folder_that_cannot_be_measured_is_a_build_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "environment"
    folder.mkdir()
    (folder / "data.bin").write_bytes(b"x")

    class RefusingOs:
        """The os module, except that reading a file's size is refused."""

        def __getattr__(self, name: str) -> Any:
            return getattr(os, name)

        @staticmethod
        def lstat(path: Any) -> Any:
            raise PermissionError(f"planted: permission denied: {path}")

    monkeypatch.setattr(environments, "os", RefusingOs())
    with pytest.raises(environments.BuildError, match="planted: permission denied"):
        environments.disk_usage(folder)


def test_a_failed_build_leaves_no_marker(tmp_path: Path) -> None:
    pip, cache = _cache(tmp_path)
    pip.failing.add("install")
    with pytest.raises(environments.BuildError, match="planted failure of install"):
        cache.prepare(_wanted())
    assert not (cache.root / _wanted().directory / environments.MARKER).exists()
    pip.failing.clear()
    assert cache.prepare(_wanted()).how == "built"


def test_pruning_removes_only_the_environments_the_pins_no_longer_need(tmp_path: Path) -> None:
    pip, cache = _cache(tmp_path)
    kept = cache.prepare(_wanted()).path
    old = cache.prepare(_wanted(version="0.2.0")).path
    interrupted = cache.root / "crewai-1.0.0-aaaaaaaaaaaa"
    interrupted.mkdir()
    (interrupted / "pyvenv.cfg").write_text("", encoding="utf-8")
    unrelated = cache.root / "notes"
    unrelated.mkdir()
    loose = cache.root / "last-run.json"
    loose.write_text("{}", encoding="utf-8")
    assert sorted(cache.prune([_wanted()])) == sorted([old, interrupted])
    assert kept.is_dir() and unrelated.is_dir() and loose.is_file()
    assert not old.exists() and not interrupted.exists()


def test_disk_usage_counts_a_hard_linked_file_once(tmp_path: Path) -> None:
    folder = tmp_path / "environment"
    folder.mkdir()
    data = folder / "data.bin"
    data.write_bytes(b"x" * 100_000)
    alone = environments.disk_usage(folder)
    assert alone > 0
    os.link(data, folder / "again.bin")
    assert environments.disk_usage(folder) == alone
    (folder / "more.bin").write_bytes(b"y" * 100_000)
    assert environments.disk_usage(folder) > alone


def test_record_rows_written_with_backslashes_are_read_as_paths(tmp_path: Path) -> None:
    """RECORD's rows use forward slashes, but a tool that wrote them on Windows may have
    used backslashes. Those rows name memvara's files all the same."""
    purelib = tmp_path / "site-packages"
    info = purelib / "memvara-0.0.1.dist-info"
    info.mkdir(parents=True)
    rows = [_record_row("memvara\\__init__.py",
                        (REPO / "memvara" / "__init__.py").read_bytes()),
            _record_row("memvara\\core.py", b"an older copy")]
    (info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert environments.stale_files(purelib) == ["memvara/core.py"]


def test_environments_over_the_disk_budget_are_named_with_their_sizes(tmp_path: Path) -> None:
    """The check the nightly tier makes once every environment is ready. An environment
    two pins share counts once."""
    def ready(version: str, size: int) -> environments.Prepared:
        wanted = _wanted(version=version)
        return environments.Prepared(wanted, tmp_path / wanted.directory,
                                     Path(sys.executable), "reused", 0.0, tmp_path, {}, size)

    small, large = ready("0.3.0", 3_000_000_000), ready("1.6.5", 6_000_000_000)
    assert environments.DISK_BUDGET == 8_000_000_000
    assert environments.over_budget([small, large]) is not None
    assert environments.over_budget([small, small], budget=6_000_000_000) is None
    message = environments.over_budget([small, large, small])
    assert message is not None
    assert "9.00 GB together, over the budget of 8.00 GB" in message, message
    assert message.splitlines()[1:] == [f"  {large.path.name}: 6000 MB",
                                        f"  {small.path.name}: 3000 MB"], message


def test_a_stale_installed_memvara_is_named_file_by_file(tmp_path: Path) -> None:
    """A reused environment gets memvara reinstalled on every run. This compares what is
    installed with this checkout, through the hashes pip wrote in RECORD."""
    purelib = tmp_path / "site-packages"
    info = purelib / "memvara-0.0.1.dist-info"
    info.mkdir(parents=True)
    rows = [_record_row("memvara/__init__.py", (REPO / "memvara" / "__init__.py").read_bytes()),
            _record_row("memvara/core.py", b"an older copy"),
            _record_row("memvara-0.0.1.dist-info/METADATA", b"metadata"),
            "memvara-0.0.1.dist-info/RECORD,,"]
    (info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert environments.stale_files(purelib) == ["memvara/core.py"]
    (info / "RECORD").write_text(rows[2] + "\n", encoding="utf-8")
    assert environments.stale_files(purelib) == ["RECORD lists no memvara file"]
    (purelib / "memvara-0.0.2.dist-info").mkdir()
    (purelib / "memvara-0.0.2.dist-info" / "RECORD").write_text("", encoding="utf-8")
    assert environments.stale_files(purelib) == [
        "2 installed copies of memvara, where there should be one"]


def test_a_command_that_cannot_start_is_a_build_error(tmp_path: Path) -> None:
    """An environment whose interpreter has gone must fail as a build error, which the
    cache answers by building the environment again. Any other exception would escape
    the session and fail every test of that environment, every night."""
    pip = environments.Pip(tmp_path / "home", python=str(tmp_path / "no-such-python"))
    with pytest.raises(environments.BuildError, match="could not start"):
        pip.create(tmp_path / "environment")
    with pytest.raises(environments.BuildError, match="could not start"):
        pip.describe(tmp_path / "environment" / "bin" / "python")


def test_an_answer_that_is_not_json_is_a_build_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Something that prints to standard output when the interpreter starts, such as a
    `.pth` file, would spoil the description. That is a build error too."""
    monkeypatch.setattr(environments, "_DESCRIBE", "print('not json')")
    pip = environments.Pip(tmp_path / "home")
    with pytest.raises(environments.BuildError, match="not JSON: not json"):
        pip.describe(Path(sys.executable))


# -- what the probe runs with ------------------------------------------------------------


def test_the_probe_environment_drops_credentials_and_switches_telemetry_off(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "LANGSMITH_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY",
                 "GITHUB_TOKEN", "SMTP_PASSWORD", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.setenv(name, "planted-secret")
    home = tmp_path / "home"
    home.mkdir()
    env = environments.probe_env(home)
    assert "planted-secret" not in env.values()
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    for name, value in environments.TELEMETRY_OFF.items():
        assert env[name] == value, name
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
                 "all_proxy"):
        assert env[name] == "http://127.0.0.1:9", name
    assert "PYTHONPATH" not in env
    assert env["HOME"] == str(home.resolve())


def test_run_probe_runs_the_checks_with_the_environments_interpreter(tmp_path: Path) -> None:
    """The glue between an environment and the probe: this interpreter stands in for the
    environment's, and one planted check stands in for a framework's."""
    checks = tmp_path / "checks_planted.py"
    checks.write_text("def check_sees_its_folder(ctx):\n    assert ctx.folder.is_dir()\n",
                      encoding="utf-8")
    prepared = environments.Prepared(_wanted(), tmp_path, Path(sys.executable), "reused",
                                     0.0, tmp_path, {}, 0)
    probed = environments.run_probe(prepared, tmp_path / "work", checks=checks)
    assert probed.exit_code == 0, probed.output
    assert probed.run.finished and probed.run.results["check_sees_its_folder"].passed
    assert probed.run.network == []


@pytest.mark.parametrize("framework", environments.FRAMEWORKS, ids=lambda f: f.name)
def test_each_checks_file_loads_the_way_the_probe_loads_it(
        framework: environments.Framework) -> None:
    """The probe loads a checks file by its path, outside any package, in an environment
    that holds one framework. A relative import, or a framework imported at the top of the
    file, would stop every check in it, and without this test only the nightly run would
    say so. The suite has no framework installed, so this loads each file the same way."""
    module = probe.load(framework.checks)
    assert probe.checks(module), f"{framework.checks.name} defines no check"


def test_a_probe_past_its_time_limit_is_stopped_with_every_process_it_started(
        tmp_path: Path) -> None:
    """A framework can start processes of its own. When the probe runs past its time
    limit, they must stop with it, or they run on after the night is over."""
    pid_file = tmp_path / "child.pid"
    checks = tmp_path / "checks_planted.py"
    checks.write_text("def check_starts_a_process_and_hangs(ctx):\n" + textwrap.indent(
        STARTS_A_PROCESS_AND_HANGS.format(pid_file=str(pid_file)), "    "), encoding="utf-8")
    prepared = environments.Prepared(_wanted(), tmp_path, Path(sys.executable), "reused",
                                     0.0, tmp_path, {}, 0)
    probed = environments.run_probe(prepared, tmp_path / "work", checks=checks, timeout=5)
    assert probed.exit_code is None, probed.output
    assert pid_file.exists(), f"the check never started its process:\n{probed.output}"
    child = int(pid_file.read_text())
    try:
        assert _poll(lambda: not _alive(child), 10), (
            f"process {child}, started by a probe that ran past its limit, still runs")
    finally:
        _stop(child)


def test_a_pip_command_past_its_time_limit_is_stopped_with_every_process_it_started(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same for pip and its build backends. Here the description of an environment
    starts a process and hangs."""
    pid_file = tmp_path / "child.pid"
    monkeypatch.setattr(environments, "_DESCRIBE",
                        STARTS_A_PROCESS_AND_HANGS.format(pid_file=str(pid_file)))
    pip = environments.Pip(tmp_path / "home", timeout=3)
    with pytest.raises(environments.BuildError, match="ran past 3 seconds"):
        pip.describe(Path(sys.executable))
    child = int(pid_file.read_text())
    try:
        assert _poll(lambda: not _alive(child), 10), (
            f"process {child}, started by a command that ran past its limit, still runs")
    finally:
        _stop(child)


# -- one run's session -------------------------------------------------------------------


def test_a_session_pins_every_framework_twice_and_prunes_the_rest(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    old = root / "crewai-1.0.0-aaaaaaaaaaaa"
    old.mkdir(parents=True)
    (old / "pyvenv.cfg").write_text("", encoding="utf-8")
    pip = FakePip(latest={**LATEST, "mem0ai": "2.0"})
    with environments.session(root, tmp_path / "work", pip) as current:  # type: ignore[arg-type]
        assert set(current.wanted) == {(f.name, pin) for f in environments.FRAMEWORKS
                                       for pin in environments.PINS}
        # FakePip answers with the pinned text; pip itself pads "0.3" to 0.3.0.
        assert current.wanted[("langchain", "floor")].version == "0.3"
        assert current.wanted[("langchain", "latest")].version == "1.6.5"
        assert current.wanted[("crewai", "floor")].version == "1.10.1"
        assert (current.wanted[("mem0", "floor")].directory
                == current.wanted[("mem0", "latest")].directory)
        assert current.removed == [old] and not old.exists()
        # A companion is resolved beside the framework at each pin, and pinned.
        assert current.wanted[("langgraph", "floor")].companions == (("langgraph", "1.2.12"),)
        current.prepare("langchain", "latest")
        assert any("langchain-core 1.6.5" in line for line in current.summary())
    assert pip.kinds().count("wheel") == 1 and pip.kinds().count("resolve") == 2
    together = [call[2:] for call in pip.calls if call[0] == "resolve with dependencies"]
    assert together == [("langgraph-checkpoint==4.1", "langgraph"),
                        ("langgraph-checkpoint==4.2.0", "langgraph")]
    record = json.loads((root / "last-run.json").read_text(encoding="utf-8"))
    assert [e["version"] for e in record["environments"]] == ["1.6.5"]
    assert record["removed"] == [old.name]


def test_the_session_resolves_exactly_the_pins_in_pins(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environments, "PINS", ("floor",))
    pip = FakePip()
    with environments.session(tmp_path / "cache", tmp_path / "work", pip) as current:  # type: ignore[arg-type]
        assert {pin for _, pin in current.wanted} == {"floor"}
    assert pip.kinds().count("resolve") == 1


def test_a_probe_that_cannot_start_is_a_build_error_the_session_remembers(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An environment whose interpreter was removed, or cannot be run, must fail as a
    build error the session remembers: every test of that environment then reports the
    one failure, rather than each starting the probe again. FakePip's interpreter is an
    empty file, which the system refuses to run."""
    started: list[str] = []
    real = environments.run_probe

    def counting(prepared: environments.Prepared, work: Path,
                 **options: Any) -> environments.ProbeRun:
        started.append(prepared.wanted.directory)
        return real(prepared, work, **options)

    monkeypatch.setattr(environments, "run_probe", counting)
    pip = FakePip()
    with environments.session(tmp_path / "cache", tmp_path / "work", pip) as current:  # type: ignore[arg-type]
        for _ in range(2):
            with pytest.raises(environments.BuildError, match="could not start"):
                current.probe("langchain", "latest")
    assert len(started) == 1


def test_a_session_prepares_each_environment_once_and_remembers_a_failure(
        tmp_path: Path) -> None:
    pip = FakePip(latest={**LATEST, "mem0ai": "2.0"})
    with environments.session(tmp_path / "cache", tmp_path / "work", pip) as current:  # type: ignore[arg-type]
        first = current.prepare("mem0", "floor")
        assert current.prepare("mem0", "latest") is first
        assert pip.kinds().count("create") == 1
        pip.failing.add("install")
        for _ in range(2):
            with pytest.raises(environments.BuildError, match="planted failure of install"):
                current.prepare("crewai", "floor")
        attempts = [call for call in pip.calls
                    if call[0] == "install" and "crewai==1.10.1" in call]
        assert len(attempts) == 1
