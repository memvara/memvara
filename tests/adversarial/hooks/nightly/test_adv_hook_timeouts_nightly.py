"""Time limits that take real seconds to check.

Recall starts no optional work after 7.5 seconds (plugin/hooks/recall.py,
`OVERALL_BUDGET_SEC`). Here a hosted endpoint answers the standing refresh's two calls
slowly enough to use that budget up, and the recall must skip its wider second read, say
so, and still answer inside its 10 seconds.

Two known bugs are pinned here, because each takes a real limit to show: a hosted endpoint
that never answers keeps session start and recall past their limits (B63, #345), and a
prompt of a few megabytes keeps recall past its limit (B66, #348). With both present, the
pins wait out about 70 seconds of limits between them.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable

import pytest

from harness import known_bugs
from harness.fakes.hosted_mcp import FakeHostedMcp
from harness.hooks import HookRunner, HookTimeout

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


# -- known bugs --------------------------------------------------------------------------

@pytest.mark.parametrize("hook", ("session_start", "recall"))
@pytest.mark.parametrize("route", ("initialize", "tools/call memory_recall"))
@known_bugs.xfail("B63")
def test_a_hosted_store_that_never_answers_costs_a_hook_no_more_than_its_limit(
        hooks: Make, hook: str, route: str) -> None:
    """The hosted client waits 6 seconds for each call and retries a call that got no
    answer once (plugin/hooks/lib/hosted.py, TIMEOUT_SEC and `_rpc`), and nothing bounds
    how many calls one hook makes. Measured on a laptop against an endpoint that never
    answers: with the handshake hung, session start ran 60 seconds and recall 36; with
    memory_recall hung, session start ran 24 and recall 12. The host kills the hook at its
    limit, so the turn gets nothing, not even the status line.

    The same hook, with a home of its own, first answers against the same endpoint while
    it still answers, so a timeout here comes from the hung call."""
    fields: dict[str, Any] = {"prompt": support.PROMPT} if hook == "recall" else {}
    with FakeHostedMcp() as fake:
        env = {"MEMVARA_API_KEY": fake.api_key, "MEMVARA_SERVER_URL": fake.serve()}
        answered = hooks("claude", env=env).run(hook, session="s", **fields)
        assert answered.exit_code == 0 and support.status_of("claude", answered.reply), (
            answered)
        fake.hang(route)
        try:
            result = hooks("claude", env=env).run(hook, session="s", **fields)
        except HookTimeout:
            raise known_bugs.Reproduced(
                f"B63: {hook} ran past its {support.LIMITS['claude'][hook]}-second limit "
                f"against a hosted endpoint whose {route!r} never answers") from None
    assert result.exit_code == 0
    assert support.status_of("claude", result.reply)


@known_bugs.xfail("B66")
def test_recall_answers_a_sixteen_megabyte_prompt_within_its_limit(
        hooks: Make, store_env: dict[str, str]) -> None:
    """The recall hook sends the whole prompt to the store as its query, and the store's
    time grows with the query: about 2.2 seconds a megabyte on a laptop, so a prompt of
    about 4.5 MB already runs past the 10-second limit. This one is 16 MB, so that a
    machine several times faster still shows the bug. The same hook first answers an
    ordinary prompt, so a timeout here comes from the prompt's size."""
    runner = hooks("claude", env=store_env)
    ordinary = runner.run("recall", session="ordinary", prompt=support.PROMPT)
    assert support.MEMORY in support.context_of("claude", ordinary.reply)
    prompt = (support.PROMPT + " ") * (16_000_000 // (len(support.PROMPT) + 1))
    try:
        result = runner.run("recall", stdin=json.dumps({"session_id": "s", "prompt": prompt}))
    except HookTimeout:
        raise known_bugs.Reproduced(
            "B66: recall ran past its 10-second limit on a 16 MB prompt") from None
    assert result.exit_code == 0
    assert support.status_of("claude", result.reply)
