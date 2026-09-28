"""plugin/hooks/lib/toml_servers.py, the reader of a Codex `config.toml` for Python 3.10.

On Python 3.11 and later the hooks read the file with `tomllib`; on 3.10 they read only
the tables and the string or inline-table values. Each document here is read both ways
where `tomllib` exists, and the part the hooks use, the MCP servers' variables, must be
the same.
"""

from __future__ import annotations

import doctest
import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from lib import toml_servers  # noqa: E402

#: Each document, and the variables of its memvara server as the hooks must read them.
DOCUMENTS = {
    "a sub-table": (
        '[mcp_servers.memvara]\ncommand = "python3"\nargs = ["-m", "memvara.server"]\n\n'
        '[mcp_servers.memvara.env]\nMEMVARA_DB = "/tmp/m.db"\nMEMVARA_USER = "tester"\n',
        {"MEMVARA_DB": "/tmp/m.db", "MEMVARA_USER": "tester"}),
    "an inline table": (
        '[mcp_servers.memvara]\ncommand = "python3"\n'
        'env = { MEMVARA_DB = "/tmp/m.db", MEMVARA_USER = "tester" }\n',
        {"MEMVARA_DB": "/tmp/m.db", "MEMVARA_USER": "tester"}),
    "settings and other servers around it": (
        'model = "o3"\napproval_policy = "never"\n[sandbox]\nmode = "workspace-write"\n'
        'writable_roots = ["/tmp", "/var"]\n\n[mcp_servers.github]\ncommand = "gh"\n'
        'env = { GH_TOKEN = "x" }\n\n[mcp_servers.memvara]\ncommand = "python3"\n'
        'startup_timeout_sec = 20\nenv = { MEMVARA_DB = "/tmp/m.db" }  # the store\n',
        {"MEMVARA_DB": "/tmp/m.db"}),
    "quoted keys, literal strings and escapes": (
        '[mcp_servers."memvara-local".env]\n"MEMVARA_DB" = \'C:\\Users\\me\\m.db\'\n'
        'MEMVARA_USER = "t\\u00e9ster \\U0001F600"\nNOTE = "a # b"\n',
        {"MEMVARA_DB": "C:\\Users\\me\\m.db", "MEMVARA_USER": "téster \U0001F600",
         "NOTE": "a # b"}),
    "an array of tables beside it": (
        '[[profiles]]\nname = "x"\n[mcp_servers.memvara]\nenv = { MEMVARA_DB = "/m" }\n',
        {"MEMVARA_DB": "/m"}),
    "arguments over several lines": (
        '[mcp_servers.memvara]\ncommand = "/venv/bin/python"\nargs = [\n  "-m",  # module\n'
        '  "memvara.server",\n]\nenv = { MEMVARA_DB = "/m" }\n',
        {"MEMVARA_DB": "/m"}),
}


def _env(document: dict | None) -> dict:
    assert isinstance(document, dict)
    servers = document["mcp_servers"]
    (name,) = [name for name in servers if "memvara" in name]
    return servers[name]["env"]


@pytest.mark.parametrize("case", DOCUMENTS)
def test_the_subset_reads_the_servers_variables(case: str) -> None:
    text, env = DOCUMENTS[case]
    assert _env(toml_servers._subset(text)) == env


def _server(document: dict | None) -> dict:
    """The memvara server's command, arguments and variables, as agentic capture uses
    them (`lib.agentic.mcp_config`)."""
    assert isinstance(document, dict)
    servers = document["mcp_servers"]
    (name,) = [name for name in servers if "memvara" in name]
    return {key: servers[name].get(key) for key in ("command", "args", "env")}


@pytest.mark.parametrize("case", DOCUMENTS)
def test_the_subset_agrees_with_tomllib(case: str) -> None:
    tomllib = pytest.importorskip("tomllib")
    text, _ = DOCUMENTS[case]
    assert _server(toml_servers._subset(text)) == _server(tomllib.loads(text))


def test_the_subset_reads_the_command_and_its_arguments() -> None:
    """Agentic capture starts the server the client names with these, so an argument list
    the reader skipped would start the interpreter with nothing to run."""
    text, _ = DOCUMENTS["arguments over several lines"]
    server = _server(toml_servers._subset(text))
    assert (server["command"], server["args"]) == ("/venv/bin/python",
                                                    ["-m", "memvara.server"])


def test_a_file_that_is_not_toml_reads_as_none_where_tomllib_reads_it() -> None:
    pytest.importorskip("tomllib")
    assert toml_servers.read("[mcp_servers.memvara\nenv = {") is None


@pytest.mark.parametrize("text", [
    '[mcp_servers.memvara.env]\nMEMVARA_DB = "\\UFFFFFFFF"\n',
    '[mcp_servers.memvara]\nenv = { MEMVARA_DB = "\\UFFFFFFFF" }\n',
    '[mcp_servers.memvara.env]\nMEMVARA_DB = "\\x41"\n',
], ids=["past the last code point", "past it, inline", "an escape JSON lacks"])
def test_an_escape_that_is_not_valid_reads_as_unreadable_without_raising(
        monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """TOML lets `\\U` take eight hex digits, and a value past U+10FFFF is not a character.
    It used to raise ValueError out of the reader, which `lib.ipc` does not catch there."""
    monkeypatch.setitem(sys.modules, "tomllib", None)  # read the way Python 3.10 reads it
    assert toml_servers.read(text) is None


def test_the_examples_in_the_docstring_still_run() -> None:
    """pytest's doctest collection covers `tests` and `memvara`, not the hooks."""
    failures, attempted = doctest.testmod(toml_servers, verbose=False)
    assert attempted > 0
    assert failures == 0
