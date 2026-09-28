"""The library and the hosted clients agree on how a session's value and a session's global
fact are read (#266, #273).

The hosted client builds no chain of scopes of its own. It sends the scope it is bound to,
and the deployment resolves the chain with this library, so the two agree as long as the
deployment runs this release. `FakeV1` answers from a real local store the way a
deployment does (`tests/harness/fakes/fake_v1.py`), so one program runs through the
library and through `RemoteMemvara`, and every answer is compared.

**One answer differs, and the test asserts the difference.** `RemoteMemvara.count()` reads
the `total` of `GET /v1/memories`, which the deployment computes from a present-tense
listing, and a present-tense listing is shadowed. So a hosted count bound to a session
leaves out the user-wide value the session's own value hides, where the library's
`count()` includes it. This predates #266: a hosted count bound to a repository already
left out a user-wide value the repository's own value hid. Making the two agree needs the
deployment to count without shadowing, which is a change to memvara-cloud.
"""

from __future__ import annotations

from typing import Any, Callable

from harness import stores
from harness.fakes.fake_v1 import FakeV1

PROJECT = "github.com/o/a"


def program(bind: Callable[..., Any]) -> dict[str, Any]:
    """The same operations on any client. `bind(**scope)` returns the client bound to the
    user and to `scope`."""
    user, session, sibling = bind(), bind(session="s1"), bind(session="s2")
    inside = bind(project=PROJECT, session="s1")
    user.remember("user", "lives_in", "Berlin")
    session.remember("user", "lives_in", "Paris")
    [own] = inside.remember("user", "likes", "tea").added

    def homes(client: Any) -> list[str]:
        return sorted(c.object for c in client.get_all() if c.predicate == "lives_in")

    return {
        "the user level reads": homes(user),
        "the session reads": homes(session),
        "a sibling session reads": homes(sibling),
        "the session inside the project reads": homes(inside),
        "the session searches": sorted(r.claim.object for r in session.search("lives in")
                                       if r.claim.predicate == "lives_in"),
        "the session counts": session.count(),
        "the session inside the project reads its own global fact":
            [c.id for c in inside.get_all() if c.predicate == "likes"] == [own.id],
        "the session inside the project explains it": inside.why(own.id) is not None,
        "the project alone reads it": [c.object for c in bind(project=PROJECT).get_all()
                                       if c.predicate == "likes"],
    }


def test_the_hosted_client_reads_scopes_as_the_library_does() -> None:
    local = stores.memory(user="alice")
    want = program(lambda **scope: local.scope(**scope))
    assert want == {
        "the user level reads": ["Berlin"],
        "the session reads": ["Paris"],
        "a sibling session reads": ["Berlin"],
        "the session inside the project reads": ["Paris"],
        "the session searches": ["Paris"],
        # Berlin, Paris, and tea: the session outside the project reads the level where
        # the same session filed its global fact from inside the project.
        "the session counts": 3,
        "the session inside the project reads its own global fact": True,
        "the session inside the project explains it": True,
        "the project alone reads it": [],
    }
    with FakeV1() as fake:
        got = program(lambda **scope: fake.remote(user="alice", **scope))
    # Paris and tea: the hosted count is the length of a shadowed listing.
    assert got.pop("the session counts") == 2
    want.pop("the session counts")
    assert got == want
