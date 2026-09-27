"""The `report` job in .github/workflows/ci.yml waits for every other job.

On a push to main, `report` opens or closes the issue in memvara/build-health from the
results of the jobs it `needs`. A job left out of that list is not waited for: `report`
can close the issue while that job is still running, and a failure in it then reaches
nobody. So the list must be every other job in the file, and a job added later must be
added to it.

The workflow is read with a small text parse, as tests/test_npm_release.py reads the
release workflows, because PyYAML is not a test dependency here.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _jobs(text: str) -> dict[str, str]:
    """Each job's id and its block of text, from the `jobs:` section of a workflow."""
    body = text.split("\njobs:\n", 1)[1]
    starts = list(re.finditer(r"^  ([a-z][a-z0-9_-]*):\s*$", body, re.M))
    return {match.group(1): body[match.end():following.start() if following else None]
            for match, following in zip(starts, starts[1:] + [None])}


def test_the_report_job_needs_every_other_job_in_the_workflow() -> None:
    jobs = _jobs(CI.read_text(encoding="utf-8"))
    needs = re.search(r"^    needs: \[([^\]]*)\]\s*$", jobs["report"], re.M)
    assert needs is not None, "report must list its needs on one line, as [a, b, c]"
    listed = [name.strip() for name in needs.group(1).split(",")]
    assert len(listed) == len(set(listed))
    assert set(listed) == set(jobs) - {"report"}
    assert len(jobs) > 5, "the text parse found too few jobs to be reading the file"


def test_the_report_job_runs_only_for_a_push_to_main() -> None:
    """release.yml calls this workflow on a tag push, where the event is also `push`, so
    the condition has to name the branch as well."""
    report = _jobs(CI.read_text(encoding="utf-8"))["report"]
    condition = re.search(r"^    if: (.+)$", report, re.M)
    assert condition is not None
    assert "github.event_name == 'push'" in condition.group(1)
    assert "github.ref == 'refs/heads/main'" in condition.group(1)
    assert "!contains(needs.*.result, 'cancelled')" in condition.group(1)
