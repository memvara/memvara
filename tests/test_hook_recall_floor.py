"""The `min_score` floor the recall hook applies, and how it degrades.

These are regressions. Every test here corresponds to a defect that was in the first draft
of this change and was found reviewing it, not to a behaviour anyone designed twice.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from lib.hosted import HostedRecall, HostedError  # noqa: E402


class Rejects:
    """A server that refuses any call carrying one named argument.

    The refusal has status 200, as the real client reports a refusal from the tool itself:
    the tool read the arguments and answered with an error inside an HTTP 200.
    """

    def __init__(self, offending: str) -> None:
        self.offending = offending
        self.seen: list[dict] = []

    def __call__(self, tool: str, args: dict) -> str:
        self.seen.append(dict(args))
        if self.offending in args:
            raise HostedError(f"no branch for {self.offending}", status=200)
        return "- a memory"


def client(monkeypatch, server) -> HostedRecall:
    made = HostedRecall("key")
    monkeypatch.setattr(made, "_call", server)
    # No network: the handshake and the `tools/list` probe answer here. The probe fails
    # (`None`), because the drops these tests are about are the fallback for a client that
    # could not read the schema. With a schema, an argument the server does not declare is
    # never sent; `tests/test_hook_recall_requests.py` counts the requests for that case.
    monkeypatch.setattr(made, "_ensure_session", lambda: True)
    monkeypatch.setattr(made, "offers", lambda tool, argument: None)
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


@pytest.mark.parametrize("status", [429, 402, 500, None])
def test_a_refusal_that_is_not_about_an_argument_is_not_sent_again(monkeypatch, status):
    """Only the tool can refuse an argument, and it answers inside an HTTP 200.

    A 429 or 402 is the allowance or the rate limit, a 500 is the server, and `None` is no
    reply at all. Each used to be retried twice more without the floor and the episodes,
    logged as "hosted rejected min_score" although nothing had rejected it.
    """
    seen: list[dict] = []

    def refuses(tool: str, args: dict) -> str:
        seen.append(dict(args))
        raise HostedError("refused", status=status)

    made = client(monkeypatch, refuses)
    with pytest.raises(HostedError):
        made.recall("q", include_episodes=True, min_score=0.29)
    assert len(seen) == 1, seen
    assert made.unfiltered is False


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


#: The scores each route's default floor was chosen from, measured on 2026-09-27 (#154).
#: Each question is scored by its top result, except the scripted session's, which is the
#: score of the memory that answers it. The numbers are the lowest score of a question the
#: store should answer and the highest score of one it should not; a floor must be above
#: the second and at or below the first.
#:
#: The local route reads through the library, whose default embedder is the hashing one.
#: The seeded store is the plugin-recall benchmark's own: 15 facts, 15 questions it should
#: answer and 22 plausible questions it cannot. The scripted session is
#: `tests/scenarios/scripted/session-recall-every-prompt.json`, whose "asks-again" prompt
#: asks about the door code and the employer at once.
#:
#: The hosted route reaches the hosted service, which embeds with all-MiniLM-L6-v2. The
#: real store is a hosted store of 2,407 claims, probed with 20 questions it should answer
#: and 8 it cannot; the probe file is private to the store and is not in this repository.
MEASURED = {
    "local": {
        "seeded store, hashing embedder": {"answerable": 0.3603, "unanswerable": 0.2346},
        "scripted session, hashing embedder": {"answerable": 0.2997},
    },
    "hosted": {
        "seeded store, all-MiniLM-L6-v2": {"answerable": 0.4704, "unanswerable": 0.2977},
        "real hosted store": {"answerable": 0.3713, "unanswerable": 0.3468},
    },
}


@pytest.mark.parametrize("route, store", [
    (route, store) for route in sorted(MEASURED) for store in sorted(MEASURED[route])])
def test_each_route_default_floor_separates_what_was_measured_on_it(route, store):
    """Each route's floor must silence every unanswerable question and keep every
    answerable one on the stores it was measured against.

    Both routes used one floor, 0.29, measured on the seeded store alone. On the real
    hosted store six of its eight unanswerable questions scored above that, so the hook
    injected memories into prompts the store knew nothing about. Raising the one floor to
    0.35 would have fixed that and dropped the scripted session's answer, which scores
    0.2997 on the hashing embedder. If you change either constant, measure again with
    `bench/hosted.py` and `python -m benchmarks.plugin_recall.calibrate`, and update these
    numbers from what you measured.
    """
    import recall as recall_hook

    floor = {"local": recall_hook.MIN_SCORE, "hosted": recall_hook.HOSTED_MIN_SCORE}[route]
    scores = MEASURED[route][store]
    if "unanswerable" in scores:
        assert scores["unanswerable"] < floor, (
            f"on the {store}, an unanswerable question scoring {scores['unanswerable']} "
            f"clears the {route} floor of {floor}, so the hook would inject memories into "
            "a prompt the store cannot answer")
    assert floor <= scores["answerable"], (
        f"on the {store}, an answerable question scoring {scores['answerable']} falls "
        f"under the {route} floor of {floor}, so the hook would stay silent on a prompt "
        "the store can answer")


def test_the_hosted_default_applies_when_nothing_is_configured(monkeypatch):
    import recall as recall_hook

    monkeypatch.delenv("MEMVARA_RECALL_MIN_SCORE", raising=False)
    assert recall_hook._min_score(recall_hook.HOSTED_MIN_SCORE) == pytest.approx(
        recall_hook.HOSTED_MIN_SCORE)


def test_a_configured_floor_overrides_both_routes(monkeypatch):
    import recall as recall_hook

    monkeypatch.setenv("MEMVARA_RECALL_MIN_SCORE", "0.4")
    assert recall_hook._min_score() == pytest.approx(0.4)
    assert recall_hook._min_score(recall_hook.HOSTED_MIN_SCORE) == pytest.approx(0.4)


class _Asked:
    """A store or hosted client that records the floor each recall was given."""

    def __init__(self) -> None:
        self.floors: list[float | None] = []

    def recall(self, query, **kwargs):
        self.floors.append(kwargs.get("min_score"))
        return ""

    def close(self) -> None:
        pass


@pytest.mark.parametrize("hosted_floor, sent", [(0.35, 0.35), (None, 0.29)])
def test_the_hosted_client_gets_the_hosted_floor(monkeypatch, tmp_path, hosted_floor, sent):
    """With no local store, `fast.recall` asks the hosted client, at `hosted_min_score`
    when it is given and at `min_score` when it is not."""
    import lib.hosted
    from lib import fast

    client = _Asked()
    monkeypatch.setattr(fast, "socket_path", lambda *a, **k: str(tmp_path / "absent.sock"))
    monkeypatch.setattr(fast, "_local_store", lambda: (None, {}, {}))
    monkeypatch.setattr(lib.hosted, "open_hosted", lambda: client)
    _, ok, _ = fast.recall("who owns billing", min_score=0.29,
                           hosted_min_score=hosted_floor, spawn=False)
    assert ok is True
    assert client.floors == [sent]


def test_a_local_store_gets_the_local_floor(monkeypatch, tmp_path):
    from lib import fast
    from lib import open as opener

    store = _Asked()
    monkeypatch.setattr(fast, "socket_path", lambda *a, **k: str(tmp_path / "absent.sock"))
    monkeypatch.setattr(opener, "open_store", lambda: store)
    _, ok, _ = fast.recall("who owns billing", min_score=0.29, hosted_min_score=0.35,
                           spawn=False)
    assert ok is True
    assert store.floors == [0.29]


@pytest.mark.parametrize("hosted, applied", [(True, 0.35), (False, 0.29)])
def test_a_daemon_applies_the_floor_for_the_backend_it_serves(hosted, applied):
    """On an install with no local store, the daemon serves the hosted client, so every
    prompt after the first reaches the hosted service through it. It must apply the hosted
    floor then, or only the first prompt of a session would get it."""
    import daemon as daemon_hook

    backend = _Asked()
    served = daemon_hook.Daemon("/tmp/unused-floor.sock", backend, hosted=hosted)
    reply = served._answer({"q": "who owns billing", "k": 2, "budget": 100,
                            "min_score": 0.29, "hosted_min_score": 0.35})
    assert reply.get("ok") is True, reply
    assert backend.floors == [applied]


def test_the_daemon_request_carries_the_hosted_floor(monkeypatch, tmp_path):
    from lib import fast

    sent = []
    monkeypatch.setattr(fast, "socket_path", lambda *a, **k: str(tmp_path / "d.sock"))
    monkeypatch.setattr(fast, "send", lambda path, request, timeout: sent.append(request)
                        or json.dumps({"ok": True, "text": ""}))
    _, ok, _ = fast.recall("who owns billing", min_score=0.29, hosted_min_score=0.35,
                           spawn=False)
    assert ok is True
    assert (sent[0]["min_score"], sent[0]["hosted_min_score"]) == (0.29, 0.35)


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

    # A real path with nothing listening, on every platform. This is what exposed the
    # Windows defect fixed alongside: `send()` built an `AF_UNIX` socket before its own
    # `try`, so on Windows the attribute error escaped `fast.recall` entirely and every
    # prompt was reported as `recall failed`.
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


def test_recall_survives_a_platform_with_no_unix_sockets(monkeypatch, tmp_path):
    """Windows has no `AF_UNIX`, and that must cost the daemon, not the recall.

    `send()` built its socket before its own `try`, so the `AttributeError` was not one of
    the failures that collapse to `None`: it escaped `send`, escaped `fast.recall`, and
    landed in `recall.py`'s catch-all, which reports `recall failed`. Every prompt on
    Windows, silently, for as long as the daemon has existed -- the failure was logged
    honestly and to a channel that made it look like an unreachable store.
    """
    import socket as socket_module

    from lib import fast
    from lib import open as opener

    class Backend:
        def recall(self, query, **kwargs):
            return f"- {query}"

    monkeypatch.delattr(socket_module, "AF_UNIX", raising=False)
    monkeypatch.setattr(fast, "socket_path", lambda *a, **k: str(tmp_path / "absent.sock"))
    monkeypatch.setattr(opener, "open_store", lambda: Backend())

    text, ok, _ = fast.recall("who owns billing", spawn=False)
    assert ok is True, "a platform without unix sockets still has an in-process route"
    assert "who owns billing" in text


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
