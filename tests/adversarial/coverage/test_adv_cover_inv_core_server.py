"""Invariants of the MCP server's startup that no older test checked.

The server is launched by an MCP client, and the only thing that client shows its user
when the server will not start is what the server wrote to standard error. So each test
here starts the server the way `python -m memvara.server` does, through `cli.main`, with
the environment as its only input, and reads the exit status and the two output streams.
The invariants are in the "Invariants and assumptions" section of
`docs/claude/mcp-server.md`.
"""

from __future__ import annotations

import io
from typing import Mapping

import pytest

from memvara.server import cli
from memvara.server.config import ServerConfig, build_memvara


def _start(env: Mapping[str, str]) -> tuple[int, str, str]:
    """Start the server with `env` and an empty stdin; return its exit status, what it
    wrote to stdout and what it wrote to stderr."""
    out, err = io.StringIO(), io.StringIO()
    status = cli.main([], env=dict(env), stdin=io.StringIO(""), stdout=out, stderr=err)
    return status, out.getvalue(), err.getvalue()


_CLOUD = {"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k"}


@pytest.mark.covers("inv:MS2")
@pytest.mark.parametrize("variable, value", [
    ("MEMVARA_LLM", "anthropic"),
    ("MEMVARA_LLM", "openai"),
    ("MEMVARA_EMBEDDER", "local"),
    ("MEMVARA_EMBEDDER", "hashing:256"),
])
def test_cloud_mode_refuses_to_start_with_a_local_model_or_embedder(
        variable: str, value: str) -> None:
    """docs/claude/mcp-server.md: "Cloud mode refuses `MEMVARA_LLM` and
    `MEMVARA_EMBEDDER`." Extraction and embedding run inside the deployment, so a value
    set here would never be used, and ignoring it silently would tell the operator
    something false.

    So the server must not start: it exits with status 2, writes nothing to stdout, and
    names the variable on stderr. The same environment without the variable does start
    a cloud client, which shows that the variable, and nothing else, is what is refused.
    """
    status, out, err = _start({**_CLOUD, variable: value})
    assert status == 2
    assert out == ""
    assert f"{variable}={value!r} does not apply under MEMVARA_MODE=cloud" in err, err

    from memvara.remote.api import RemoteMemvara
    memory = build_memvara(ServerConfig.from_env(_CLOUD))
    try:
        assert isinstance(memory, RemoteMemvara)
    finally:
        memory.close()


#: One wrong environment per kind of setting the server reads, with the variable the
#: refusal must name and a piece of the valid value it must show.
_WRONG = [
    ({"MEMVARA_MODE": "remote"}, "MEMVARA_MODE", "'local' or 'cloud'"),
    ({}, "MEMVARA_DB", '"MEMVARA_DB": "/absolute/path/to/memory.db"'),
    ({"MEMVARA_DB": ":memory:", "MEMVARA_EMBEDDER": "bogus"}, "MEMVARA_EMBEDDER",
     "'hashing', 'hashing:<dim>', 'local', 'local:<model>' or 'auto'"),
    ({"MEMVARA_DB": ":memory:", "MEMVARA_EMBEDDER": "hashing:wide"}, "MEMVARA_EMBEDDER",
     "positive integer"),
    ({"MEMVARA_DB": ":memory:", "MEMVARA_LLM": "gpt"}, "MEMVARA_LLM",
     "'none', 'anthropic' or 'openai'"),
    ({"MEMVARA_DB": ":memory:", "MEMVARA_READ_ONLY": "maybe"}, "MEMVARA_READ_ONLY",
     "1, on, true, yes or 0, false, no, off"),
    ({"MEMVARA_DB": ":memory:", "MEMVARA_FEATURE_DOCUMENTS": "sometimes"},
     "MEMVARA_FEATURE_DOCUMENTS", "1, on, true, yes or 0, false, no, off"),
    ({"MEMVARA_DB": ":memory:", "MEMVARA_FEATURE_NOPE": "1"}, "MEMVARA_FEATURE_NOPE",
     "The features are"),
    ({"MEMVARA_DB": ":memory:", "MEMVARA_READ_W_GRAPH": "heavy"}, "MEMVARA_READ_W_GRAPH",
     "finite number of zero or more"),
    ({"MEMVARA_MODE": "cloud"}, "MEMVARA_API_KEY", "memvara-mcp login"),
]


@pytest.mark.covers("inv:MS3")
@pytest.mark.parametrize("env, variable, valid", _WRONG,
                         ids=[f"{variable}-{i}" for i, (_, variable, _v) in enumerate(_WRONG)])
def test_a_configuration_error_is_a_refusal_that_names_the_variable_and_the_fix(
        env: dict[str, str], variable: str, valid: str) -> None:
    """docs/claude/mcp-server.md: "A configuration error is a refusal that names the
    fix." The client that launches the server shows its user nothing but the startup
    failure, so the message has to say which variable was wrong and what a valid value
    looks like.

    For each wrong environment the server must exit with status 2 before it speaks the
    protocol, write nothing to stdout, and write one message to stderr, not a traceback,
    that names the variable and shows a valid value.
    """
    status, out, err = _start(env)
    assert status == 2
    assert out == ""
    assert err.startswith("memvara-mcp: ") and "Traceback" not in err, err
    assert variable in err, err
    assert valid in err, err
