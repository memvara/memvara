"""The `min_score` floor the recall hook applies, and how it degrades.

These are regressions. Every test here corresponds to a defect that was in the first draft
of this change and was found reviewing it, not to a behaviour anyone designed twice.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from lib.hosted import HostedRecall, HostedError  # noqa: E402


class Rejects:
    """A server that refuses any call carrying one named argument."""

    def __init__(self, offending: str) -> None:
        self.offending = offending
        self.seen: list[dict] = []

    def __call__(self, tool: str, args: dict) -> str:
        self.seen.append(dict(args))
        if self.offending in args:
            raise HostedError(f"no branch for {self.offending}")
        return "- a memory"


def client(monkeypatch, server) -> HostedRecall:
    made = HostedRecall("key")
    monkeypatch.setattr(made, "_call", server)
    return made


def test_a_rejected_include_episodes_still_degrades_when_a_floor_is_also_set(monkeypatch):
    """The regression this file exists for.

    `min_score` was added as the first branch of the `except` and returned from inside it,
    so a call carrying both arguments and rejected because of `include_episodes` retried
    with the episodes still attached, failed again, and propagated -- making the older
    `include_episodes` fallback unreachable for the one call site that uses it. Since the
    floor now defaults to non-zero, that call site is the episode-widening retry on every
    hosted prompt.
    """
    server = Rejects("include_episodes")
    text = client(monkeypatch, server).recall(
        "q", include_episodes=True, min_score=0.29)
    assert text.strip() == "- a memory"
    # Dropped in order, cumulatively: floor first, episodes second.
    assert [sorted(set(a) & {"min_score", "include_episodes"}) for a in server.seen] == [
        ["include_episodes", "min_score"], ["include_episodes"], []]


def test_a_rejected_floor_is_dropped_and_recorded(monkeypatch):
    server = Rejects("min_score")
    made = client(monkeypatch, server)
    text = made.recall("q", min_score=0.29)
    assert text.strip() == "- a memory"
    assert made.unfiltered is True, (
        "a hosted store that cannot filter must be distinguishable from one that did")


def test_unfiltered_is_readable_before_any_call():
    """It was only ever created inside `recall()`, so reading it first raised."""
    assert HostedRecall("key").unfiltered is False


def test_a_supported_floor_is_left_alone(monkeypatch):
    server = Rejects("nothing-is-rejected")
    made = client(monkeypatch, server)
    made.recall("q", min_score=0.29)
    assert made.unfiltered is False
    assert server.seen[0]["min_score"] == 0.29


@pytest.mark.parametrize("raw, expected", [
    ("0", 0.0),          # honoured: restores the old unfiltered behaviour
    ("0.5", 0.5),
    ("5", 1.0),          # clamped: above 1.0 filters everything on the local route
    ("-3", 0.0),
    ("banana", 0.29),    # unparseable falls back to the default rather than to no floor
])
def test_the_configured_floor_is_clamped_to_the_range_scores_occupy(
        monkeypatch, raw, expected):
    import recall as recall_hook

    monkeypatch.setenv("MEMVARA_RECALL_MIN_SCORE", raw)
    assert recall_hook._min_score() == pytest.approx(expected)


def test_the_default_applies_when_nothing_is_configured(monkeypatch):
    import recall as recall_hook

    monkeypatch.delenv("MEMVARA_RECALL_MIN_SCORE", raising=False)
    assert recall_hook._min_score() == pytest.approx(recall_hook.MIN_SCORE)


class OlderBackend:
    """A store whose `recall()` predates `min_score`, which is most of them."""

    def recall(self, query, k=6, budget=700, header=None,
               include_episodes=False, memory_types=None):
        return f"- {query}"


def test_the_daemon_does_not_send_a_floor_it_was_not_given():
    """The daemon and the direct path must call one backend identically.

    `lib.fast.recall` adds `min_score` only when it is set. The daemon was written to add
    it always, on the reasoning that `0.0` filters nothing -- which is true of the value
    and false of the call: a backend whose signature predates the argument raises
    `TypeError`, so the daemon route returned nothing at all while the direct route
    answered normally. `claude-memvara`'s route-parity test caught it as
    `(True, None, None)`, and this asserts the same thing where the code lives.
    """
    import daemon as daemon_hook

    served = daemon_hook.Daemon("/tmp/unused-parity.sock", OlderBackend())
    reply = served._answer({"q": "who owns billing", "k": 2, "budget": 100})
    assert reply.get("ok") is True, (
        f"a backend without min_score must still be answerable: {reply}")
    assert "who owns billing" in reply.get("text", "")


@pytest.mark.parametrize("floor", [0.0, 0.29])
def test_both_routes_build_the_same_call_over_one_backend(monkeypatch, tmp_path, floor):
    """The parity invariant itself, asserted where the code lives.

    The test above pins one argument by name, which is enough to stop *this* regression
    coming back and not enough to stop the next one: add a new optional argument to either
    call site and every test here still passes, while the divergence surfaces only when
    `claude-memvara` next vendors the tree and its route-parity test fails on the sync PR
    -- after the change has already merged here. That is precisely the sequence that
    produced this fix.

    So both routes are driven over one backend and their text compared. The backend records
    the keyword arguments it was handed, and the two records must match exactly: it is the
    *call* that has to agree, and two routes can return identical text while disagreeing
    about what they asked for, right up until a backend cares.
    """
    import daemon as daemon_hook
    from lib import fast
    from lib import open as opener

    class Recorder:
        def __init__(self):
            self.calls = []

        def recall(self, query, **kwargs):
            self.calls.append(dict(kwargs))
            return f"- {query} ({sorted(kwargs)})"

    direct_backend, daemon_backend = Recorder(), Recorder()
    query, args = "who owns billing", {"k": 3, "budget": 200}

    # No daemon listening, so `fast.recall` takes the in-process route to `open_store`.
    monkeypatch.setattr(fast, "socket_path", lambda *a, **k: str(tmp_path / "absent.sock"))
    monkeypatch.setattr(opener, "open_store", lambda: direct_backend)
    direct_text, ok, _ = fast.recall(query, min_score=floor, spawn=False, **args)
    assert ok is True

    daemon_reply = daemon_hook.Daemon(str(tmp_path / "unused.sock"),
                                      daemon_backend)._answer({"q": query, "min_score": floor,
                                                               **args})
    assert daemon_reply.get("ok") is True, daemon_reply

    assert daemon_reply["text"] == direct_text, "the two routes disagree on the answer"
    assert daemon_backend.calls == direct_backend.calls, (
        "the two routes disagree on what they asked the backend: "
        f"daemon={daemon_backend.calls} direct={direct_backend.calls}")


def test_the_standing_interval_still_has_its_own_documentation():
    """The floor's comment block was appended to this constant's, leaving it undocumented
    and attributing its measured rationale to the floor instead."""
    source = (HOOKS / "recall.py").read_text()
    intro = "#: How often a running session re-checks whether its standing preferences"
    assert intro in source
    after_intro = source[source.index(intro):]
    between = after_intro[:after_intro.index("STANDING_REFRESH_SECONDS = 15 * 60")]
    assert "MIN_SCORE" not in between, (
        "MIN_SCORE has been moved back inside the standing-refresh comment block")
    assert "222 ms" in between, "the interval's own measured rationale went missing"
