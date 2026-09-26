"""The framework tests' environments: made ready once per run, and summarised at its end."""

from __future__ import annotations

from typing import Iterator

import pytest

from .. import environments

_SESSION = pytest.StashKey[environments.Session]()


@pytest.fixture(scope="session")
def frameworks(tmp_path_factory: pytest.TempPathFactory,
               pytestconfig: pytest.Config) -> Iterator[environments.Session]:
    """Resolve every framework's two pins, remove the environments they no longer need,
    and make each environment ready the first time a test asks for it."""
    work = tmp_path_factory.mktemp("frameworks")
    with environments.session(environments.CACHE, work) as session:
        pytestconfig.stash[_SESSION] = session
        yield session


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, exitstatus: int,
                            config: pytest.Config) -> None:
    """Print what each environment resolved to, its size on disk and how long it took."""
    session = config.stash.get(_SESSION, None)
    if session is None:
        return
    terminalreporter.section("framework environments")
    for line in session.summary():
        terminalreporter.write_line(line)
