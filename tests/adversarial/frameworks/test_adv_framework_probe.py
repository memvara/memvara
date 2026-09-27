"""The probe that runs a framework's checks inside that framework's environment.

These tests run the probe with this interpreter, on checks planted in a temporary file,
so they need no framework. Nothing here reaches the network: the probe's guard refuses
each attempt inside the process, before anything is sent.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from . import environments, probe

#: The probe's script, run the way the nightly tests run it.
PROBE = environments.PROBE

PLANTED = """
    import atexit
    import socket


    def check_passes(ctx):
        assert ctx.folder.is_dir()


    def check_fails(ctx):
        raise AssertionError("planted failure")


    def check_connects(ctx):
        socket.create_connection(("127.0.0.1", 9), timeout=1)


    def check_looks_up_a_name(ctx):
        socket.getaddrinfo("example.invalid", 443)


    def check_reaches_out_after_the_last_check(ctx):
        atexit.register(socket.create_connection, ("127.0.0.1", 9), 1)


    def helper_that_is_not_a_check(ctx):
        raise AssertionError("a helper is never run")
"""

HANGS = """
    import time


    def check_hangs(ctx):
        time.sleep(60)


    def check_after_the_hang(ctx):
        pass
"""

#: A check that passes but leaves a thread running, so the interpreter never finishes
#: shutting down: it waits for every thread that is not a daemon before it runs the exit
#: handlers.
HANGS_AT_EXIT = """
    import threading
    import time


    def check_leaves_a_thread_running(ctx):
        threading.Thread(target=time.sleep, args=(30,), daemon=False).start()
"""

#: A checks file whose own import never finishes.
HANGS_WHILE_LOADING = """
    import time

    time.sleep(30)


    def check_never_reached(ctx):
        pass
"""

#: A checks file whose own import fails, the way one that imports its framework at the
#: top would fail in the suite's interpreter.
FAILS_WHILE_LOADING = """
    raise ImportError("planted: no module named crewai")


    def check_never_reached(ctx):
        pass
"""

#: Runs the probe in-process with an import finder that makes `import memvara` hang.
#: That import is the first thing `run()` does, before any check is loaded. Formatted
#: with the probe's folder, the checks file and the report.
HANGS_IMPORTING_MEMVARA = """
import sys
import time


class Hang:
    def find_spec(self, name, path=None, target=None):
        if name == "memvara":
            time.sleep(30)
        return None


sys.meta_path.insert(0, Hang())
sys.path.insert(0, {folder!r})
import probe

sys.exit(probe.main([{checks!r}, {report!r}, "--check-seconds", "0.5"]))
"""

Probed = tuple[probe.Run, "subprocess.CompletedProcess[str]"]


def _probe(folder: Path, source: str, *options: str, timeout: float = 120) -> Probed:
    """Run the probe on the checks in `source`, with this interpreter."""
    checks = folder / "checks_planted.py"
    checks.write_text(textwrap.dedent(source), encoding="utf-8")
    report = folder / "report.jsonl"
    home = folder / "home"
    home.mkdir()
    done = subprocess.run(
        [sys.executable, "-I", "-B", str(PROBE), str(checks), str(report), *options],
        env=environments.probe_env(home), cwd=folder, capture_output=True, text=True,
        timeout=timeout)
    return probe.read(report), done


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> probe.Run:
    run, _ = _probe(tmp_path_factory.mktemp("planted"), PLANTED)
    return run


@pytest.fixture(scope="module")
def hung(tmp_path_factory: pytest.TempPathFactory) -> Probed:
    return _probe(tmp_path_factory.mktemp("hung"), HANGS, "--check-seconds", "0.5")


def _accesses(run: probe.Run, phase: str) -> list[str]:
    return [record["access"] for record in run.network if record["phase"] == phase]


def test_each_check_runs_once_in_the_order_its_file_defines_it(planted: probe.Run) -> None:
    assert list(planted.results) == [
        "check_passes", "check_fails", "check_connects", "check_looks_up_a_name",
        "check_reaches_out_after_the_last_check"]
    assert planted.finished


def test_a_failed_check_is_reported_and_the_next_one_still_runs(planted: probe.Run) -> None:
    assert planted.results["check_passes"].passed
    failed = planted.results["check_fails"]
    assert (failed.passed, failed.error_type, failed.message) == (
        False, "AssertionError", "planted failure")
    assert "planted failure" in failed.traceback
    assert "check_connects" in planted.results


def test_a_connection_is_refused_and_recorded_against_its_check(planted: probe.Run) -> None:
    result = planted.results["check_connects"]
    assert (result.passed, result.error_type) == (False, "BlockedNetworkAccess")
    assert _accesses(planted, "check_connects") == ["connect to 127.0.0.1:9"]


def test_a_name_lookup_is_refused_and_recorded_against_its_check(planted: probe.Run) -> None:
    result = planted.results["check_looks_up_a_name"]
    assert (result.passed, result.error_type) == (False, "BlockedNetworkAccess")
    assert _accesses(planted, "check_looks_up_a_name") == ["look up example.invalid"]


def test_an_attempt_after_the_last_check_is_still_recorded(planted: probe.Run) -> None:
    """Telemetry often sends from an exit handler, after every check has passed. CrewAI's
    does, when it is left on."""
    assert planted.results["check_reaches_out_after_the_last_check"].passed
    assert _accesses(planted, "exit") == ["connect to 127.0.0.1:9"]


def test_the_start_record_names_what_the_interpreter_can_import(planted: probe.Run) -> None:
    assert planted.start is not None
    assert planted.start["distributions"].get("pytest")


def test_a_check_that_hangs_is_stopped_and_named(hung: Probed) -> None:
    run, done = hung
    assert run.begun == ["check_hangs"] and run.results == {} and not run.finished
    assert done.returncode != 0 and "Timeout" in done.stderr, done.stderr
    assert "stopped while running it" in run.result("check_hangs").message
    after = run.result("check_after_the_hang")
    assert (after.passed, after.error_type) == (False, "ProbeStopped")
    assert "check_hangs" in after.message


def test_a_probe_that_hangs_after_its_last_check_is_stopped(tmp_path: Path) -> None:
    """A framework can leave a thread running that never ends. The interpreter then never
    reaches the exit handlers, where telemetry sends, so the probe must stop itself and
    exit with a failure rather than wait until the suite kills it."""
    run, done = _probe(tmp_path, HANGS_AT_EXIT, "--check-seconds", "0.5", timeout=20)
    assert run.results["check_leaves_a_thread_running"].passed and run.finished
    assert done.returncode != 0 and "Timeout" in done.stderr, done.stderr


def test_a_checks_file_that_hangs_while_loading_is_stopped(tmp_path: Path) -> None:
    run, done = _probe(tmp_path, HANGS_WHILE_LOADING, "--check-seconds", "0.5", timeout=20)
    assert run.start is not None and run.begun == [] and not run.finished
    assert done.returncode != 0 and "Timeout" in done.stderr, done.stderr
    assert "before its first check" in run.result("check_never_reached").message


def test_a_probe_that_hangs_before_it_loads_its_checks_is_stopped(tmp_path: Path) -> None:
    """The watchdog runs from the first line of `run()`. Here `import memvara`, which the
    probe does before it writes its first record, never finishes. The probe must stop
    itself and print the stack that shows where it hung, rather than wait for the suite's
    own time limit, which ends it without a stack."""
    checks = tmp_path / "checks_planted.py"
    checks.write_text("def check_never_reached(ctx):\n    pass\n", encoding="utf-8")
    report = tmp_path / "report.jsonl"
    home = tmp_path / "home"
    home.mkdir()
    code = HANGS_IMPORTING_MEMVARA.format(folder=str(PROBE.parent), checks=str(checks),
                                          report=str(report))
    done = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                          env=environments.probe_env(home), cwd=tmp_path,
                          capture_output=True, text=True, timeout=20)
    assert done.returncode != 0 and "Timeout" in done.stderr, done.stderr
    assert "find_spec" in done.stderr, done.stderr
    run = probe.read(report)
    assert run.start is None and not run.finished


def test_a_checks_file_that_fails_to_load_gives_every_check_the_cause(tmp_path: Path) -> None:
    """A checks file that raises while it loads runs no check. The probe records the
    failure once, naming the file and the exception, and every check reports that cause
    instead of only saying that it never ran."""
    run, done = _probe(tmp_path, FAILS_WHILE_LOADING)
    assert done.returncode == 1, done.stderr
    assert run.start is not None and run.begun == [] and not run.finished
    assert run.load_error is not None
    assert Path(run.load_error.check).name == "checks_planted.py"
    assert (run.load_error.error_type, run.load_error.message) == (
        "ImportError", "planted: no module named crewai")
    result = run.result("check_never_reached")
    assert (result.passed, result.error_type) == (False, "ChecksNotLoaded")
    assert "checks_planted.py" in result.message, result.message
    assert "ImportError: planted: no module named crewai" in result.message, result.message
    assert "planted: no module named crewai" in result.traceback


def test_network_access_is_recognised_in_each_form() -> None:
    """Connections and datagrams to an address count, and so do lookups of a name. What
    stays on the machine does not: a unix socket, a numeric address, localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp, \
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        assert probe.network_access("socket.connect", (tcp, ("192.0.2.1", 443))) == (
            "connect to 192.0.2.1:443")
        assert probe.network_access("socket.sendto", (udp, ("192.0.2.1", 53))) == (
            "send to 192.0.2.1:53")
        assert probe.network_access("socket.bind", (tcp, ("127.0.0.1", 0))) is None
    if hasattr(socket, "AF_UNIX"):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix:
            assert probe.network_access("socket.connect", (unix, "/tmp/d.sock")) is None
    # CPython raises the event socket.gethostbyname for gethostbyname_ex as well.
    for event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
        assert probe.network_access(event, ("pypi.org", 443)) == "look up pypi.org"
        assert probe.network_access(event, (b"pypi.org", 443)) == "look up pypi.org"
        for local in ("127.0.0.1", "::1", "localhost", "", None):
            assert probe.network_access(event, (local, 443)) is None, (event, local)
    # A numeric address is answered without a name server, except in a reverse lookup,
    # which asks one what the address is called.
    assert probe.network_access("socket.getaddrinfo", ("192.0.2.1", 443)) is None
    assert probe.network_access("socket.gethostbyaddr", ("192.0.2.1",)) == (
        "look up 192.0.2.1")
    assert probe.network_access("socket.getnameinfo", (("192.0.2.1", 80),)) == (
        "look up 192.0.2.1")
    assert probe.network_access("socket.getnameinfo", (("127.0.0.1", 80),)) is None
    assert probe.network_access("open", ("/etc/hosts", "r", 0)) is None


def test_a_report_cut_short_is_read_as_far_as_it_goes(tmp_path: Path) -> None:
    lines = [
        json.dumps({"kind": "start", "distributions": {}}),
        json.dumps({"kind": "begin", "check": "check_a"}),
        json.dumps({"kind": "result", "check": "check_a", "passed": True, "seconds": 0.1}),
        json.dumps({"kind": "begin", "check": "check_b"}),
        '{"kind": "result", "check": "che',
    ]
    report = tmp_path / "report.jsonl"
    report.write_text("\n".join(lines), encoding="utf-8")
    run = probe.read(report)
    assert run.results["check_a"].passed and not run.finished
    assert run.begun == ["check_a", "check_b"]
    assert run.result("check_b").error_type == "ProbeStopped"
    missing = probe.read(tmp_path / "missing.jsonl").result("check_a")
    assert "before its first check" in missing.message


def test_a_check_a_finished_run_never_reported_is_named_as_unknown(tmp_path: Path) -> None:
    """A run that finished ran every check its file defines, so a check with no result is
    one the file does not define, such as a renamed check that a pin still names."""
    report = tmp_path / "report.jsonl"
    report.write_text("\n".join(json.dumps(record) for record in (
        {"kind": "begin", "check": "check_a"},
        {"kind": "result", "check": "check_a", "passed": True},
        {"kind": "end"})), encoding="utf-8")
    result = probe.read(report).result("check_renamed")
    assert (result.passed, result.error_type) == (False, "ProbeStopped")
    assert "the probe finished" in result.message and "check_a" not in result.message


def test_only_the_check_functions_a_file_defines_are_listed(tmp_path: Path) -> None:
    source = tmp_path / "checks_listed.py"
    source.write_text(textwrap.dedent("""
        from os.path import join as check_imported


        def check_b(ctx):
            pass


        def helper(ctx):
            pass


        def check_a(ctx):
            pass


        check_constant = 3
    """), encoding="utf-8")
    module = probe.load(source)
    assert [name for name, _ in probe.checks(module)] == ["check_b", "check_a"]
