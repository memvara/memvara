"""Time limits that take real seconds to check.

Recall starts no optional work after 7.5 seconds (plugin/hooks/recall.py,
`OVERALL_BUDGET_SEC`). Here a hosted endpoint answers the standing refresh's two calls
slowly enough to use that budget up, and the recall must skip its wider second read, say
so, and still answer inside its 10 seconds.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness.fakes.hosted_mcp import FakeHostedMcp
from harness.hooks import HookRunner

from .. import support

Make = Callable[..., HookRunner]


def test_recall_stops_its_optional_work_at_its_real_budget(hooks: Make) -> None:
    with FakeHostedMcp() as fake:
        for call in ("tools/call memory_standing", "tools/call memory_since"):
            fake.delay(call, 3.9)
        runner = hooks("claude", env={"MEMVARA_API_KEY": fake.api_key,
                                      "MEMVARA_SERVER_URL": fake.serve()})
        result = runner.run("recall", session="s", prompt=support.UNRELATED)
    assert result.exit_code == 0
    assert result.elapsed > 7.5, "the slow calls did not use the budget up"
    assert "skipped=episode widen, budget exhausted" in result.log("recall")
    assert support.status_of("claude", result.reply) == (
        "⋈ Memvara · no matching memories")


@pytest.mark.parametrize("host", ("copilot", "cursor"))
def test_capture_frees_the_turn_while_its_extractor_hangs(
        hooks: Make, tmp_path: pathlib.Path, host: str) -> None:
    support.check_capture_frees_the_turn(hooks, tmp_path, host)
