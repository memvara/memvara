"""Fixtures for the scripted sessions and the tool-surface tests."""

from __future__ import annotations

import pytest

from . import switches


@pytest.fixture(scope="session")
def surface_template(tmp_path_factory: pytest.TempPathFactory) -> switches.Template:
    """The seeded store every real server of the tool-surface tests starts from. It is
    built once per run and copied for each server, so no run sees another's writes."""
    return switches.Template.build(tmp_path_factory.mktemp("surface-template"))
