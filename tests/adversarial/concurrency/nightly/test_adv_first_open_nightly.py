"""Real processes opening one new store at the same moment, 60 times over. Every one must
open it. Before the fix for #281, about one round in five had a process that failed at
once with "database is locked", so 60 clean rounds happened by chance about one night in
a million."""

from __future__ import annotations

import pathlib

from harness.crash import Child, CrashHarnessError
from harness.stdio import kill_all

ROUNDS = 60
PROCESSES = 3


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
    assert failures == []
