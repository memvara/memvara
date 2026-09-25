"""Hypothesis runs under the profile of the selected tier."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness.skips import explained
from harness.tiers import HYPOTHESIS_PROFILE_FOR


def test_hypothesis_runs_the_profile_of_the_selected_tier(
        request: pytest.FixtureRequest) -> None:
    expected = settings.get_profile(HYPOTHESIS_PROFILE_FOR[request.config.getoption("--tier")])
    assert settings.default.max_examples == expected.max_examples
    assert settings.default.derandomize == expected.derandomize


def test_the_fast_profile_is_repeatable_and_keeps_no_database() -> None:
    fast = settings.get_profile("memvara-fast")
    assert fast.derandomize is True
    assert fast.database is None


@given(st.text())
def test_the_skip_ledger_answers_for_any_reason_text(reason: str) -> None:
    assert explained(reason) in (True, False)
