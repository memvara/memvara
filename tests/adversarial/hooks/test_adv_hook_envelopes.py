"""Every hook on every host answers in the envelope that host reads.

Each host's record in plugin/hooks/hosts/<id>.py says what its client was measured to
read off a hook's stdout. `support.SHAPES` and `support.EVENTS` restate those
measurements by hand, so a record or a renderer that drifts from them fails here, rather
than shipping a hook whose reply the client ignores. Every run sends the payload that
host sends, with the extra keys its record lists (`support.host_payload`), against a
store that holds `support.MEMORY`.

The runs are made once for the module, side by side, and each test reads its own.

The restatement is deliberate, because a test that read the record would agree with any
change to it. So the last tests here compare `support.HOSTS`, `support.EVENTS`,
`support.SHAPES` and `support.DETACHES` with the records directly, and when a record
changes, the failure names the table that has gone stale.
"""

from __future__ import annotations

import sys
from typing import Any, Iterator

import pytest

from harness import stores
from harness.fakes.cli import NO_FAKES
from harness.hooks import host_ids, host_record

from . import support

#: A string whose words do not matter to the client, in `shape`.
TEXT = "<text>"

#: The hosts that fire an event for recall. Cursor fires none.
RECALL_HOSTS = tuple(host for host in support.HOSTS if "recall" in support.EVENTS[host])


def shape(value: Any) -> Any:
    """`value` with every string replaced by TEXT, except the words a client matches
    exactly: an event name, and an approval's verdict."""
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items()}
    if isinstance(value, str) and value not in support.EXACT_WORDS:
        return TEXT
    return value


def expected(host: str, hook: str) -> dict[str, Any]:
    """The shape of `hook`'s reply on `host`, built from the tables in `support`."""
    nested, status, context, verdict, reason = support.SHAPES[host]
    fields: dict[str, Any] = ({verdict: "allow", reason: TEXT} if hook == "approve"
                              else {context: TEXT})
    body = ({"hookSpecificOutput": {"hookEventName": support.EVENTS[host][hook], **fields}}
            if nested else fields)
    if status is not None and hook != "approve":
        body = {status: TEXT, **body}
    return body


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[support.Runs]:
    """Every hook on every host, run once against a store that holds MEMORY. Capture runs
    against the fake agent CLIs, with a store of its own for each host."""
    base = tmp_path_factory.mktemp("envelopes")
    work = base / "work"
    work.mkdir()
    env = support.store_env(support.make_store(base / "memory.db"))
    jobs: support.Jobs = {}
    stores_by_host = {}
    with support.runner_factory(work) as make:
        for host in support.HOSTS:
            def payload(hook: str, **fields: Any) -> str:
                return support.host_json(host, hook, session="envelopes", cwd=work,
                                         **fields)

            jobs["session_start", host] = support.job(
                make(host, env=env), "session_start", stdin=payload("session_start"))
            if host in RECALL_HOSTS:
                jobs["recall", host] = support.job(
                    make(host, env=env), "recall",
                    stdin=payload("recall", prompt=support.PROMPT))
            jobs["approve", host] = support.job(
                make(host, env=env), "approve",
                stdin=payload("approve", tool_name=support.tool_name(host, "memory_search"),
                              tool_input={"query": "where does the user live"}))
            jobs["approve a write", host] = support.job(
                make(host, env=env), "approve",
                stdin=payload("approve", tool_name=support.tool_name(host, "memory_forget"),
                              tool_input={"claim_id": "cl_0"}))
            if sys.platform != "win32":
                clis = support.script_clis(base / f"clis-{host}", host)
                db = stores_by_host[host] = support.make_store(base / f"capture-{host}.db",
                                                               memory=False)
                transcript = support.write_transcript(
                    host, base / f"transcript-{host}.jsonl",
                    [(support.USER_TURN, support.ASSISTANT_TURN)])
                jobs["capture", host] = support.job(
                    make(host, env=support.store_env(db), stubs=clis),
                    "capture", stdin=payload("capture", transcript_path=str(transcript)))
        # Cursor has no recall event, so its runner is given the Claude-shaped payload
        # and a limit of its own.
        jobs["recall", "cursor"] = support.job(
            make("cursor", env=env), "recall", timeout=10,
            stdin=support.host_json("claude", "recall", session="envelopes", cwd=work,
                                    prompt=support.PROMPT))
        yield support.Runs(support.run_all(jobs), stores_by_host)


def _status_line(host: str, reply: dict[str, Any] | None) -> None:
    status = support.status_of(host, reply)
    if support.SHAPES[host][1] is None:
        assert status is None
    else:
        assert status is not None and status.startswith("⋈ Memvara · "), status


@pytest.mark.parametrize("host", support.HOSTS)
def test_session_start_answers_in_the_envelope_the_host_reads(runs: support.Runs,
                                                              host: str) -> None:
    result = runs["session_start", host]
    assert (result.exit_code, result.stderr) == (0, "")
    assert shape(result.reply) == expected(host, "session_start")
    assert support.MEMORY in support.context_of(host, result.reply)
    _status_line(host, result.reply)


@pytest.mark.parametrize("host", RECALL_HOSTS)
def test_recall_answers_in_the_envelope_the_host_reads(runs: support.Runs,
                                                       host: str) -> None:
    result = runs["recall", host]
    assert (result.exit_code, result.stderr) == (0, "")
    assert shape(result.reply) == expected(host, "recall")
    assert support.MEMORY in support.context_of(host, result.reply)
    _status_line(host, result.reply)


def test_cursor_has_no_recall_and_the_dispatcher_says_it_skipped_it(
        runs: support.Runs) -> None:
    """Cursor never fires the event that would carry recall (hosts/cursor.py), so run.py
    skips the hook and says so in the log, rather than answering in some other shape."""
    result = runs["recall", "cursor"]
    assert (result.exit_code, result.stdout, result.stderr) == (0, "", "")
    assert result.log("hooks") == ("skipped=cursor has no event for recall",)


@pytest.mark.parametrize("host", support.HOSTS)
def test_approve_answers_in_the_envelope_the_host_reads(runs: support.Runs,
                                                        host: str) -> None:
    result = runs["approve", host]
    assert (result.exit_code, result.stderr) == (0, "")
    assert shape(result.reply) == expected(host, "approve")


@pytest.mark.parametrize("host", support.HOSTS)
def test_approve_says_nothing_about_a_tool_that_writes(runs: support.Runs,
                                                       host: str) -> None:
    """Silence leaves the client to ask the person, which is what a write must do."""
    result = runs["approve a write", host]
    assert (result.exit_code, result.stdout, result.stderr) == (0, "", "")


@pytest.mark.parametrize("host", support.HOSTS)
def test_capture_prints_nothing_and_stores_the_fact_its_turn_states(
        runs: support.Runs, host: str) -> None:
    """Every host discards what capture prints, because it runs in the background there
    (async, detached, or not awaited), so capture must print nothing at all. On Codex,
    Copilot and Cursor, run.py hands it to a child in a new session."""
    if sys.platform == "win32":
        pytest.skip(NO_FAKES)
    result = runs["capture", host]
    assert (result.exit_code, result.stdout, result.stderr) == (0, "", "")
    assert (result.detached_pid is not None) is (host in support.DETACHES)
    assert any("stored=1" in line for line in result.log("capture")), result.logs
    with stores.file(runs.stores[host]) as mem:
        facts = [claim.object for claim in mem.scope(user=support.USER).get_all()]
    assert facts == ["Lisbon"]


# -- the tables in support.py, against the host records ----------------------------------

def test_support_hosts_are_the_hosts_with_a_record() -> None:
    assert support.HOSTS == host_ids(), "support.HOSTS is stale"


@pytest.mark.parametrize("host", support.HOSTS)
def test_support_events_are_the_ones_the_host_record_names(host: str) -> None:
    assert support.EVENTS[host] == dict(host_record(host).events), (
        f"support.EVENTS is stale for {host}")


@pytest.mark.parametrize("host", support.HOSTS)
def test_support_shapes_are_the_keys_the_host_record_names(host: str) -> None:
    """SHAPES says whether the reply is nested, its status key (None where the host shows
    no status line), its context key, and an approval's verdict and reason keys."""
    record = host_record(host)
    nested, status, context, verdict, reason = support.SHAPES[host]
    assert (("nested" if nested else "flat"), status or "", context, verdict, reason) == (
        record.envelope, record.status_key, record.context_key,
        record.approve.decision_key, record.approve.reason_key), (
        f"support.SHAPES is stale for {host}")


def test_support_detaches_names_the_hosts_whose_record_detaches_capture() -> None:
    detaching = frozenset(host for host in host_ids() if host_record(host).detach_capture)
    assert support.DETACHES == detaching, "support.DETACHES is stale"
