"""Two real MCP servers on one store file, as two agent sessions on one machine would run.

They take turns writing, then each reads back everything both wrote. No acknowledged
write may be lost or stored twice, and the file must be intact once both have exited.
The writes use a predicate that holds many values, so no write ends another and the
final state is known exactly. The store is created before the servers start, because two
processes creating one store at once meet #281, which `test_adv_first_open.py` pins.
"""

from __future__ import annotations

import pathlib

from harness import stores
from harness.invariants import check_store_integrity
from harness.stdio import McpProcess

HERBS = (("almond", "basil", "cedar", "dill", "elder", "fennel", "ginger", "hazel", "iris",
          "juniper"),
         ("kale", "lemon", "mint", "nutmeg", "olive", "pepper", "quince", "rosemary",
          "sage", "thyme"))


def test_two_servers_on_one_file_lose_nothing_and_read_each_others_writes(
        tmp_path: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    stores.file(db).close()
    homes = [tmp_path / "one", tmp_path / "two"]
    for home in homes:
        home.mkdir()
    with McpProcess(db, home=homes[0]) as one, McpProcess(db, home=homes[1]) as two:
        servers = (one, two)
        for server in servers:
            server.initialize()
        for i in range(10):
            for server, herbs in zip(servers, HERBS):
                reply = server.call("memory_remember", subject="user", predicate="likes",
                                    object=herbs[i])
                assert not reply.is_error, reply.text
        for server in servers:
            for herb in HERBS[0] + HERBS[1]:
                reply = server.call("memory_search", query=herb, k=3)
                assert not reply.is_error and herb in reply.text, (
                    f"a server cannot read {herb!r}: {reply.text}")
    with stores.file(db) as mem:
        likes = sorted(c.object for c in mem.get_all(user="tester"))
    assert likes == sorted(HERBS[0] + HERBS[1])
    assert check_store_integrity(db) == []
