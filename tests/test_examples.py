"""The examples run, and print what the documentation says they print.

An example in a README is a claim, and the ones under `examples/` are the first code a
developer reads. Nothing checked them before this file: they could import a method that
had been renamed, or print a value the surrounding prose contradicted, and the suite would
stay green — which is the failure mode `CLAUDE.md` describes for documentation generally,
with the added sting that this is the code somebody copies.

Each example is run **in a subprocess**, as a reader would run it, rather than imported
and called. That is the difference between checking that a function works and checking
that the file works: a missing `if __name__ == "__main__"`, a top-level import that only
resolves from inside the package, or a `SystemExit` are all invisible to an import-and-call
test and fatal to the person following the README.

**An example has to run on every interpreter and platform CI covers, and this suite is
what makes that concrete.** The first run of it found two failures, both invisible to
anyone writing this code on Linux with a current Python, and both fatal to a real reader:

* `coding_agent.py` built its vocabulary with `load_all_specs`, which parses TOML with
  `tomllib` — and `tomllib` arrives in 3.11, while `requires-python` promises 3.10. The
  example now declares the same predicates as `PredicateSpec`s, which is all a pack file
  is.
* The demo printed a box-drawing rule and died on Windows, where `sys.stdout` defaults to
  cp1252. It now sets its own output encoding, and `run()` below decodes with the same
  one rather than with the parent's locale.

Neither would have been caught by reading the files, and neither is a CI artifact: both
are exactly what a person on that interpreter or that platform would have hit. That is
the argument for running examples rather than only linting them.

The demo's transcript is asserted against a golden file rather than against strings in
here, so the demo, its README and this test cannot drift into three different versions of
one thing. Regenerate it with:

    python3 examples/temporal_memory_demo/demo.py --fast \\
      > examples/temporal_memory_demo/expected-output.txt
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def run(script: Path, *args: str) -> str:
    """Run one example the way a reader would, and return what it printed.

    `cwd=ROOT` and `PYTHONPATH` explicitly removed: the examples import `memvara` and
    nothing else, so they must work against the *installed* package. An example that only
    runs with the repository on the path is an example that fails for everyone who
    pip-installed.

    Removed rather than merely not set, because a subprocess inherits the environment: on
    a developer machine that exports `PYTHONPATH=.` — which `bench/` and `demo/` both
    need — the repository would be on the path anyway and this test would quietly stop
    asking its question.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # `encoding="utf-8"` rather than bare `text=True`: the demo writes box-drawing
    # characters, and `text=True` decodes with the *parent's* locale — cp1252 on Windows,
    # which would turn a correct transcript into a mismatch against the golden file.
    proc = subprocess.run([sys.executable, str(script), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert proc.returncode == 0, (
        f"{script.relative_to(ROOT)} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc.stdout


# -- 1. the temporal example, which is the one the README leads with -------------------


@pytest.fixture(scope="module")
def temporal_output() -> str:
    return run(EXAMPLES / "temporal_memory.py")


def test_the_temporal_example_answers_one_question_three_different_ways(
        temporal_output: str) -> None:
    """The headline claim: three instants, three different correct answers.

    Asserted as ordered pairs rather than as substrings anywhere in the output, because
    "New York appears somewhere" would pass on an example that printed the same answer
    three times.
    """
    lines = [line.strip() for line in temporal_output.splitlines() if line.strip()]
    answers = [lines[i + 1] for i, line in enumerate(lines) if line.startswith("Where d")]
    assert answers == ["New York", "London", "Berlin"], (
        f"expected now/March/January to differ; got {answers}")


def test_the_temporal_example_prints_a_timeline_with_two_ended_values(
        temporal_output: str) -> None:
    """Nothing was overwritten, which is what makes the three answers possible."""
    assert "Berlin    2026-01-10 -> 2026-03-15  [ended]" in temporal_output
    assert "London    2026-03-15 -> 2026-06-02  [ended]" in temporal_output
    assert "New York  2026-06-02 -> now         [live]" in temporal_output


def test_the_temporal_example_narrates_the_past_instant(temporal_output: str) -> None:
    """`ask()` is the read that composes the two clocks into a sentence."""
    assert "Alice lives_in: London." in temporal_output
    assert "It stopped being true 2026-06-02." in temporal_output
    assert "Now: New York." in temporal_output


# -- 2. the coding-agent example ------------------------------------------------------


@pytest.fixture(scope="module")
def coding_output() -> str:
    return run(EXAMPLES / "coding_agent.py")


def test_the_coding_agent_example_supersedes_rather_than_accumulating(
        coding_output: str) -> None:
    """`auth_strategy` is declared `Cardinality.ONE`, and the declaration is the point.

    Without it the predicate takes the unregistered default — multi-valued — and the
    OAuth write lands *beside* the API-keys claim instead of closing it. Both outcomes
    print a plausible-looking timeline, so the assertion is on the state.
    """
    assert "API keys                    2026-02-03 -> 2026-06-12  [ended]" in coding_output
    assert "OAuth 2.0 client credentials 2026-06-12 -> now         [live]" in coding_output
    assert "changed on 2026-06-12" in coding_output


def test_the_coding_agent_example_answers_why_with_the_actual_turn(
        coding_output: str) -> None:
    """`why()` returns the message, not a paraphrase of it — and what it replaced."""
    assert "Decision: migrate service-to-service auth from API keys to OAuth 2.0" in coding_output
    assert "recorded by: api (user)" in coding_output
    assert "it replaced: checkout-service auth strategy API keys" in coding_output


def test_the_coding_agent_example_answers_a_past_instant(coding_output: str) -> None:
    """On 1 April the answer was API keys, and it still is when you ask about April."""
    section = coding_output.split("What would we have said on 1 April 2026?")[1]
    assert section.strip().splitlines()[0].strip() == "API keys"


def test_the_coding_agent_example_keeps_the_decision_beside_the_slot(
        coding_output: str) -> None:
    """`decided` is multi-valued: a later decision does not make an earlier one untrue.

    A `ONE` cardinality here would have every new decision silently end the last, which is
    the mistake the pack's own comments warn about.
    """
    section = coding_output.split("What decisions has this service recorded?")[1]
    assert "[2026-06-12] migrate service-to-service auth to OAuth 2.0 client credentials" \
        in section


# -- 3. the demo, against its golden transcript ---------------------------------------


def test_the_demo_prints_exactly_what_its_golden_file_says() -> None:
    """`--fast` removes the pauses and nothing else, so the transcript is stable.

    Byte-for-byte rather than line-by-line: the beats are separated by blank lines and the
    rules are a fixed width, so a change in spacing is a change to what a recording will
    look like, and this is the only thing that would notice.
    """
    golden = (EXAMPLES / "temporal_memory_demo" / "expected-output.txt").read_text(
        encoding="utf-8")
    assert run(EXAMPLES / "temporal_memory_demo" / "demo.py", "--fast") == golden, (
        "the demo's output no longer matches expected-output.txt. If the change was "
        "intended, regenerate the golden file in the same commit — the command is in "
        "this module's docstring and in the demo's README.")


def test_the_demos_beat_table_matches_the_schedule_it_actually_runs() -> None:
    """`README.md` beside the demo publishes the six beats and their boundaries.

    Those numbers are how somebody plans a recording and writes a voiceover, so a table
    that no longer matches `BEATS` sends them to record 90 seconds of something that
    takes 40. Read out of the module rather than restated here, so this test cannot
    become a third version of the same numbers.
    """
    import importlib.util

    demo_py = EXAMPLES / "temporal_memory_demo" / "demo.py"
    spec = importlib.util.spec_from_file_location("_demo_under_test", demo_py)
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    page = (EXAMPLES / "temporal_memory_demo" / "README.md").read_text(encoding="utf-8")
    starts = [0.0] + list(demo.BEATS.values())[:-1]
    for start, end in zip(starts, demo.BEATS.values()):
        # The table writes them with an en dash, e.g. `| 65–80s |`.
        row = f"| {start:g}\u2013{end:g}s |"
        assert row in page, f"README.md has no beat row {row!r}"

    assert list(demo.BEATS.values())[-1] == 90.0, (
        "the demo is described everywhere as 90 seconds; if that changed, the README, "
        "this module's docstring and the repository README all say otherwise")


def test_the_demo_shows_what_the_live_value_replaced() -> None:
    """The provenance beat. `why()` answers it from the supersession chain alone.

    Worth asserting separately from the golden file because it is the one line in the
    demo that is not about time: the record knows *what* this value displaced, which is
    the difference between a note and a record.
    """
    output = run(EXAMPLES / "temporal_memory_demo" / "demo.py", "--fast")
    assert "it replaced:  London (ended)" in output
    assert "written by:   api (user)" in output


def test_the_demo_ends_on_the_positioning_line() -> None:
    """The last beat is the name and the claim, and it is the frame a recording loops on."""
    output = run(EXAMPLES / "temporal_memory_demo" / "demo.py", "--fast")
    assert "Memvara — bitemporal memory for AI agents." in output
    assert "Know what was true. Know when it was true. Know why you believe it." in output


# -- every example, in general --------------------------------------------------------


@pytest.mark.parametrize(
    "script",
    sorted(p for p in EXAMPLES.rglob("*.py")),
    ids=lambda p: p.relative_to(EXAMPLES).as_posix())
def test_every_example_is_listed_in_the_examples_index(script: Path) -> None:
    """An example nobody links to is an example nobody runs, including this suite.

    The index is where a reader picks one, so a file that is not named there has been
    added and forgotten — which is how a directory of examples turns into a directory of
    half-maintained scripts.
    """
    index = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    rel = script.relative_to(EXAMPLES).as_posix()

    # The path, or the directory that holds it *written as a path* — `temporal_memory_demo/`
    # is listed as a directory rather than by its `demo.py`. The bare directory *name* is
    # not enough: `script.parent.name` is "examples" for anything at the top level, the
    # word "examples" is all over this index, and the assertion passed for every possible
    # file. A guard that cannot fail is worse than none, because the suite reports it green.
    parent = script.parent.relative_to(EXAMPLES).as_posix()
    # `parent` is "." for a file directly under examples/, and "./" occurs inside every
    # "../docs/..." link in the index — so the directory fallback applies to real
    # subdirectories only. Getting this wrong is how the first version of this assertion
    # passed for every conceivable file.
    listed = rel in index or (parent != "." and f"{parent}/" in index)
    assert listed, (
        f"{rel} is not mentioned in examples/README.md. Add it to the index — an example "
        "nobody links to is an example nobody runs.")
