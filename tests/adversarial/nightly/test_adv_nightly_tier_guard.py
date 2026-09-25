"""Proof that the nightly folder is collected only when a run selects it."""

import pytest


def test_this_file_runs_only_when_the_nightly_tier_is_selected(
        request: pytest.FixtureRequest) -> None:
    assert request.config.getoption("--tier") in ("nightly", "weekly")
