"""Real processes opening one new store at the same moment, 60 times over. Every one must
open it. Today about one round in five has a process that fails at once with "database is
locked" (#281); 60 rounds make a clean night by chance about one in a million."""

from __future__ import annotations

import pathlib

from harness import known_bugs
from harness.crash import Child, CrashHarnessError
from harness.stdio import kill_all

ROUNDS = 60
PROCESSES = 3


@known_bugs.xfail("B14")
def test_processes_opening_a_new_store_at_once_all_open_it(tmp_path: pathlib.Path) -> None:
    failures: list[str] = []
    for round_ in range(ROUNDS):
        db = tmp_path / f"r{round_}.db"
        children = []
        try:
            for n in range(PROCESSES):
                home = tmp_path / f"home{round_}-{n}"
                home.mkdir()
                children.append(Child({"db": str(db), "user": "u1", "setup": [],
                                       "point": "before-action", "hold": True,
                                       "action": ["open", {}]}, home=home, timeout=60))
            for child in children:
                child.wait_for("POINT before-action")
            for child in children:
                child.release()
            for child in children:
                try:
                    child.wait_for("DONE")
                except CrashHarnessError as exc:
                    failures.append(f"round {round_}: {exc}".splitlines()[0])
        finally:
            kill_all(child for child in children if child.proc.poll() is None)
    if failures and all("database is locked" in f for f in failures):
        raise known_bugs.Reproduced(f"{len(failures)} opens failed: {failures[:3]}")
    assert failures == []
